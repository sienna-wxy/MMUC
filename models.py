from torch.autograd import Variable
from embedding import *
from collections import OrderedDict
import torch
import torch.nn.functional as F


class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(2 * embed_size, num_hidden1)),
            ('bn', nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden1, num_hidden2)),
            ('bn', nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape  # [batch_size, few, 2, emb_dim]
        x = inputs.contiguous().view(size[0], size[1], -1)  # [batch_size, few, 2*emb_dim]
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)  # [batch_size, out_size]

        return x.view(size[0], 1, 1, self.out_size)  # [batch_size, 1, 1, out_size]


class EmbeddingLearner(nn.Module):  # TransE
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num, neg_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


class uncertainty_generalized_loss(nn.Module):
    """ cite 'GTransE: Generalizing Translation-Based Model on Uncertain Knowledge Graph Embedding'
        https://link.springer.com/chapter/10.1007/978-3-030-39878-1_16 """
    def __init__(self, margin=1, gamma=1, device='cpu'):
        super(uncertainty_generalized_loss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.device = device

    def forward(self, p_score, n_score, confidence):
        zero = torch.Tensor([0]).to(self.device)
        y = torch.Tensor([1]).to(self.device)
        loss = torch.max(zero, pow(confidence, self.gamma) * self.margin - y * (p_score - n_score))
        return loss.mean()


class Confidence_Learner(nn.Module):
    """ Logistic Function """
    def __init__(self):
        super(Confidence_Learner, self).__init__()
        self.w = nn.Parameter(torch.Tensor([1.0]))
        self.b = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        return torch.sigmoid(self.w * x + self.b)


class Matcher(nn.Module):
    def __init__(self, input_dim, process_step=4, device='cpu'):
        super(Matcher, self).__init__()
        self.input_dim = input_dim
        self.process_step = process_step
        self.process = nn.LSTMCell(input_dim, 2 * input_dim)
        self.device = device

    def forward(self, batch_support, batch_support_confidence, batch_query, query_score, rel):
        """
            batch_support: (batch_size, few, 2, entity_dim)
            batch_support_confidence: (batch_size, few, 1)
            query: (batch_size, query, 2, entity_dim)
            query_score: (batch_size, query, 1)
            rel: (batch_size, 1, relation_dim)
        return: (batch_size, query)
        """
        batch_size = batch_query.shape[0]
        few = batch_support.shape[1]
        query_num = batch_query.shape[1]
        support_head = batch_support[:, :, 0, :]  # (batch_size, few, entity_dim)
        support_tail = batch_support[:, :, 1, :]  # (batch_size, few, entity_dim)
        query_head = batch_query[:, :, 0, :]
        query_tail = batch_query[:, :, 1, :]

        support_relation = rel.expand(batch_size, few, -1)  # (batch_size, few, relation_dim)
        query_relation = rel.expand(batch_size, query_num, -1)
        batch_support = support_head + support_relation + support_tail  # (batch_size, few, 1*entity_dim)
        batch_query = query_head + query_relation + query_tail

        batch_query_similarity = torch.zeros(batch_size, query_num).to(self.device)  # (batch_size, query_num)
        for batch in range(batch_size):
            # keep the highest confidence triple in support
            if batch_size == 1:
                support_confidence = batch_support_confidence
            else:
                support_confidence = batch_support_confidence[batch]  # (few,)
            _, indices = torch.sort(support_confidence, descending=True)
            support = batch_support[batch][indices[0]].view(1, -1)  # (1, 3*entity_dim)
            query = batch_query[batch]  # (query_num, 3*entity_dim)

            assert support.size()[1] == query.size()[1]

            if self.process_step == 0:
                return query

            h_r = Variable(torch.zeros(query_num, 2 * self.input_dim)).to(self.device)
            c = Variable(torch.zeros(query_num, 2 * self.input_dim)).to(self.device)
            for step in range(self.process_step):
                h_r_, c = self.process(query, (h_r, c))
                h = query + h_r_[:, :self.input_dim]  # (batch_size, query_dim)
                attn = F.softmax(torch.matmul(h, support.t()), dim=1)
                r = torch.matmul(attn, support)  # (batch_size, support_dim)
                h_r = torch.cat((h, r), dim=1)
            batch_query_similarity[batch] = torch.matmul(h, support.t()).squeeze()

        return query_score * batch_query_similarity


class MMUC(nn.Module):
    def __init__(self, dataset, parameter):
        super(MMUC, self).__init__()
        self.device = parameter['device']
        self.delta = parameter['delta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.embedding = Embedding(dataset, parameter)

        self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=self.embed_dim, num_hidden1=500,
                                                    num_hidden2=200, out_size=self.embed_dim, dropout_p=self.dropout_p)

        self.embedding_learner = EmbeddingLearner()
        self.rel_q_sharing = dict()

        self.mse_loss = nn.MSELoss()

        self.gamma = parameter['gamma']
        self.un_ge_loss = uncertainty_generalized_loss(margin=self.margin, gamma=self.gamma, device=self.device)

        self.matcher = Matcher(self.embed_dim * 1, 2, self.device)
        self.confidence_learner = Confidence_Learner()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :], negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :], negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, iseval=False, curr_rel=''):
        support, support_negative, query, negative = [self.embedding(t) for t in task]

        # support set, triple confidence, Float [batch_size, few]
        support_confidence_score = [[[float(t[3])] for t in query] for query in task[0]]  # 取出support中triples的置信度
        support_confidence_score = torch.FloatTensor(support_confidence_score).to(self.device).squeeze()

        # query set, triple confidence, Float [batch_size, query]
        query_confidence_score = [[[float(t[3])] for t in query] for query in task[2]]  # 取出query的置信度
        query_confidence_score = torch.FloatTensor(query_confidence_score).to(self.device).squeeze()

        num_few = support.shape[1]  # K
        num_support_negative = support_negative.shape[1]  # 1 negative triple
        num_query = query.shape[1]
        num_query_negative = negative.shape[1]

        rel = self.relation_learner(support)
        rel.retain_grad()

        rel_s = rel.expand(-1, num_few + num_support_negative, -1, -1)

        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)
            p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, num_few, num_support_negative)
            self.zero_grad()
            loss = self.un_ge_loss(p_score, n_score, support_confidence_score)
            loss.backward(retain_graph=True)
            grad_meta = rel.grad
            rel_q = rel - self.delta * grad_meta

            self.rel_q_sharing[curr_rel] = rel_q

        rel_match = rel_q.squeeze(1)
        rel_q = rel_q.expand(-1, num_query + num_query_negative, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [batch, nq+nn, 1, ent_dim]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_query, num_query_negative)

        uncertainty_similarity = self.matcher(support, support_confidence_score, query, p_score, rel_match)
        predict_score = self.confidence_learner(uncertainty_similarity)

        return p_score, n_score, predict_score.squeeze(), query_confidence_score
