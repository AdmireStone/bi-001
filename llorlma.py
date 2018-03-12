#coding: utf8
'''
Local low rank logistic MA
16:46
'''
import numpy as np
import random
class LLORLMA(object):

    def __init__(self, k, sim_h, cfix = 5, K1 = 5, K2 = 5,
                 num_factors = 10, theta = 0.02, lambda_d = 0.01, lambda_t = 0.01,
                 alpha = 0.1, beta = 0.1, max_iter=100):
        '''
        :param drug_sm:
        :param target_sm:
        '''
        self.k = k   # the num of anchor points
        self.h = float(sim_h) # the threshold of the neighbors range
        self.cfix = int(cfix)  # importance level for positive observations
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.num_factors = int(num_factors)
        self.theta = float(theta)
        self.lambda_d = float(lambda_d)
        self.lambda_t = float(lambda_t)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_iter = int(max_iter)


    def distance(self, a, b, smMat):
        return smMat[a, b]

    def fix_model(self, W, intMat,drugMat, targetMat, seed):
        '''

        :param W:
        :param intmat:
        :param seed:
        :return:
        '''
        self.local_models = []
        self.num_drugs, self.num_targets = intMat.shape
        # select anchor points
        x, y = np.where(W > 0)
        idx = range(len(x))
        random.shuffle(idx)
        train_x = x[idx]
        train_y = y[idx]
        self.anchor_points_x = train_x[:self.k]
        self.anchor_points_y = train_y[:self.k]
        self.ones = np.ones((self.num_drugs, self.num_targets))
        self.intMat = self.cfix*intMat*W
        # 这里有问题，因为参加训练drug和targets
        # 这里只选择交互为1，一个假设就是所有参加训练的drugs和targets至少参加了一次交互
        # 也就是说 set(temp_x.tolist()) == set(x)
        temp_x, temp_y = np.where(self.intMat > 0)
        self.train_drugs, self.train_targets = set(temp_x.tolist()), set(temp_y.tolist())
        self.intMat1 = (self.cfix-1)*intMat*W + self.ones
        self.construct_neighborhood(drugMat, targetMat)
        # todo paralle
        for a_t, b_t in zip(self.anchor_points_x, self.anchor_points_y):
            # for each anchor points, calculate the neighbor drugs of a_t, and the neighbor items of b_t
            self.N_a = drugMat[a_t, :] * (drugMat[a_t,:] > self.h)
            self.N_b = targetMat[b_t, :] * (targetMat[b_t,:] > self.h)
            U_t, V_t = self.ALGD_optimization(seed)
            self.local_models.append(np.dot(U_t, V_t.T))

    def ALGD_optimization(self,seed=None):
        if seed is None:
            self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_targets, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_targets, self.num_factors))
        dg_sum = np.zeros((self.num_drugs, self.U.shape[1]))
        tg_sum = np.zeros((self.num_targets, self.V.shape[1]))
        last_log = self.log_likelihood()
        for t in range(self.max_iter):
            dg = self.deriv(True)
            # dg = self.N_a.reshape(self.num_drugs,-1)*dg
            dg_sum += np.square(dg)
            vec_step_size = self.theta / np.sqrt(dg_sum)
            self.U += vec_step_size * dg
            tg = self.deriv(False)
            # tg = self.N_b.reshape(self.num_targets,-1)*tg
            tg_sum += np.square(tg)
            vec_step_size = self.theta / np.sqrt(tg_sum)
            self.V += vec_step_size * tg
            curr_log = self.log_likelihood()
            delta_log = (curr_log-last_log)/abs(last_log)
            if abs(delta_log) < 1e-5:
                break
            last_log = curr_log
        return self.U, self.V

    def remove_overflow(self,data_mat):
            x,y = np.where(data_mat > 100)
            data_mat[x,y] = 100
            x, y = np.where(data_mat < -100)
            data_mat[x, y] = -100
            return data_mat


    def log_likelihood(self):
        # import warnings
        # warnings.filterwarnings('error')

        loglik = 0
        A = np.dot(self.U, self.V.T)
        A = self.nbMat * A
        B = A * self.intMat
        loglik += np.sum(B)
        A = self.remove_overflow(A)
        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = self.intMat1 * A

        loglik -= np.sum(A)
        loglik -= 0.5 * self.lambda_d * np.sum(np.square(self.U))+0.5 * self.lambda_t * np.sum(np.square(self.V))

        return loglik

    def deriv(self, drug):
        if drug:
            vec_deriv = np.dot(self.intMat, self.V)
        else:
            vec_deriv = np.dot(self.intMat.T, self.U)
        A = np.dot(self.U, self.V.T)
        A = np.exp(A)
        A /= (A + self.ones)
        A = self.intMat1 * A
        if drug:
            vec_deriv -= np.dot(A, self.V)
            vec_deriv -= self.lambda_d*self.U+self.alpha*np.dot(self.DL, self.U)
        else:
            vec_deriv -= np.dot(A.T, self.U)
            vec_deriv -= self.lambda_t*self.V+self.beta*np.dot(self.TL, self.V)
        return vec_deriv

    def get_capped_model(self, d, t, drugMat, targetMat):
        N_a = drugMat[d, :] * (drugMat[d, :] > self.h)
        N_b = targetMat[t, :] * (targetMat[t, :] > self.h)
        nbMat = np.dot(N_a.reshape(-1, 1), N_b.reshape(1, -1))
        denominator = np.sum(nbMat[self.anchor_points_x, self.anchor_points_y])
        assert  denominator > 0
        double_capped_model = None
        idx=0
        for x,y,local_model in zip(self.anchor_points_x,self.anchor_points_y,self.local_models):
            if double_capped_model == None:
                double_capped_model = (nbMat[x,y]) * local_model
            else:
                double_capped_model += (nbMat[x,y]) * local_model
        return double_capped_model

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.U, self.V.T)
        B = A * self.intMat
        loglik += np.sum(B)
        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = self.intMat1 * A
        loglik -= np.sum(A)
        loglik -= 0.5 * self.lambda_d * np.sum(np.square(self.U))+0.5 * self.lambda_t * np.sum(np.square(self.V))
        loglik -= 0.5 * self.alpha * np.sum(np.diag((np.dot(self.U.T, self.DL)).dot(self.U)))
        loglik -= 0.5 * self.beta * np.sum(np.diag((np.dot(self.V.T, self.TL)).dot(self.V)))
        return loglik

    def construct_neighborhood(self, drugMat, targetMat):
        self.dsMat = drugMat - np.diag(np.diag(drugMat))
        self.tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1 = self.get_nearest_neighbors(self.dsMat, self.K1)
            self.DL = self.laplacian_matrix(S1)
            S2 = self.get_nearest_neighbors(self.tsMat, self.K1)
            self.TL = self.laplacian_matrix(S2)
        else:
            self.DL = self.laplacian_matrix(self.dsMat)
            self.TL = self.laplacian_matrix(self.tsMat)

    def laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5*(np.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L

    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in xrange(m):
            ii = np.argsort(S[i, :])[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    def get_local_models_weight_dstb(self, d, t, drugMat, targetMat):
        '''
        获取local model的权重分布.
        (d,t)与各个锚点的相似度加权平均.
        注意: 这里暂时没有使用 self.h 来筛选
        :param d:
        :param t:
        :return:
        '''
        weights = []
        for a_t, b_t in zip(self.anchor_points_x, self.anchor_points_y):
            s_drug = drugMat[a_t, d]
            s_target = targetMat[b_t, t]
            weights.append(s_drug*s_target)

        weights /= np.sum(weights)
        return np.array(weights)

    def get_local_models_score(self,d, t):
        local_models_score = []
        for local_model in self.local_models:
            val = local_model[d, t]
            local_models_score.append(1./ (1 + np.exp(-val)))
        return np.array(local_models_score)


    def evaluation(self, test_data, test_label, drugMat, targetMat):
        from sklearn.metrics import precision_recall_curve, roc_curve
        from sklearn.metrics import auc
        scores = []
        # temp_labels=[]
        # idx = 0
        for d, t in test_data:
            # 1. 获得每个local model的权重
            # 2. 获取各个local model 的g概率预测
            # 2. 加权组合，求平均
            weights = self.get_local_models_weight_dstb(d,t,drugMat,targetMat)
            scores_models = self.get_local_models_score(d, t)
            score = (np.dot(weights.reshape(1,-1),scores_models.reshape(-1, 1))).ravel()[0]
            scores.append(score)

        prec, rec, thr = precision_recall_curve(test_label, np.array(scores))
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, np.array(scores))
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val


if __name__=="__main__":
    import time
    from functions import *
    seeds = [7771, 8367, 22, 1812, 4659]
    folds = 5
    cvs = 1
    debug = 1
    num_anchor = 30
    sim_h = 0.01
    llorma_model = LLORLMA(k=num_anchor, sim_h=sim_h, cfix=5, num_factors=100, lambda_d=0.125, lambda_t=0.125, theta=0.5, max_iter=100)

    for dataset in ["nr", "gpcr", "ic", "e"]:
        print "**********training dataset:{0}*************".format(dataset)
        intMat, drugMat, targetMat = load_data_from_file(dataset, "/Users/dong/Desktop/bi/datasets/")
        print "avg_drug_sim=%f, avg_target_sim=%f" % (np.average(drugMat), np.average(targetMat)),
        if cvs == 1:  # CV setting CVS1
            X, D, T, cv = intMat, drugMat, targetMat, 1
        if cvs == 2:  # CV setting CVS2
            X, D, T, cv = intMat, drugMat, targetMat, 0
        if cvs == 3:  # CV setting CVS3
            X, D, T, cv = intMat.T, targetMat, drugMat, 0
        cv_data = cross_validation(X, seeds, cv,num=folds)

        aupr, auc_list = [], []
        tic = time.clock()
        for seed in seeds:
            # print "**********seed:{0}********".format(seed)
            fold_count = 0
            for W, test_data, test_label in cv_data[seed]:
                # print "Fold:{0}".format(fold_count)
                fold_count = fold_count + 1
                model = llorma_model
                model.fix_model(W, intMat, drugMat, targetMat, seed)
                aupr_val, auc_val = model.evaluation(test_data, test_label,drugMat,targetMat)
                aupr.append(aupr_val)
                auc_list.append(auc_val)
            # if debug:
            #     break
        aupr_vec = np.array(aupr, dtype=np.float64)
        auc_vec = np.array(auc_list, dtype=np.float64)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print "\n auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" % (
        auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)




