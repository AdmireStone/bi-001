'''
Steffen Rendle "BPR: Bayesian Personalized Ranking From implicit Feedback" 2009
'''
import numpy as np
import random
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
class BPR(object):
    def __init__(self, K=10, lambda_u=0.01, lambda_i=0.01, lambda_j=0.01, learn_rate=0.02,max_iter=100,
                 loss_type=0,neighbor_reg=0, lambda_d=0.01, lambda_t=0.01, alpha=0.01, beta=0.01,cfix=5):
        '''

        :param K:
        :param lambda_u: the regularization of users
        :param lambda_i: the regularization of positivie items
        :param lambda_j: the regularization of negative  items
        :param max_iter:
        '''
        self.cfix = int(cfix)  # importance level for positive observations
        self.K = K
        self.K1 = int(5)
        self.K2 = int(5)
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.lambda_d = float(lambda_d)
        self.lambda_t = float(lambda_t)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eta = learn_rate
        self.max_iter = max_iter
        self.loss_type = loss_type
        self.neighbor_reg = neighbor_reg

    def sample_negtive_sample(self,positive_targets,all_targets):
        '''
        :param positive_targets:
        :param all_targets:
        :return:
        '''
        j = random.choice(all_targets)
        while j in positive_targets:
            j = random.randint(0, self.num_targets - 1)
        return j

    def draw_train_sample(self, tr_drugs,tr_targets,data):
        '''
        :param drugs: the index of drug items in observed interaction matrix. shape (n,)
        :param targets: the index of target items in observed interaction matrix (n,)
        :param data: the complete interaction matrix
        :return:
        '''

        assert len(tr_drugs)==len(tr_targets)
        idxs=range(len(tr_drugs))
        random.shuffle(idxs)
        tr_drugs=tr_drugs[idxs]
        tr_targets=tr_targets[idxs]

        train_data = data[tr_drugs,tr_targets]
        tr_drugs_set = np.unique(tr_drugs)
        tr_targets_set = np.unique(tr_targets)

        tr_pos_idx = np.where(train_data > 0)
        tr_pos_durgs = tr_drugs[tr_pos_idx]
        tr_pos_targets = tr_targets[tr_pos_idx]

        for idx in np.ravel(tr_pos_idx):
            # <u,i> should be 1 in intmat
            u=tr_drugs[idx]
            i=tr_targets[idx]
            assert data[u, i] == 1

            # if a user interacts with all the targets,it has no negtive number
            u_tr_pos_targets = np.where(data[u, :] > 0)[0]
            debug_a = np.sum(data[u,:]>0)
            if len(u_tr_pos_targets)==self.num_targets:
                print "warning: skip user={0}".format(u)
                continue
            #NOTE: the negative targets is not chose from all the non-observed items of specific user,but only from the
            # all the observed items which have no interaction with the certain user.s
            j=self.sample_negtive_sample(u_tr_pos_targets,tr_targets_set)
            assert data[u, j] == 0,'<u,j>=<{0},{1}>'.format(u,j)
            yield  u,i,j

    def draw_train_sample_special(self,tr_drugs,tr_targets,data):
        # assert len(drugs) == len(targets)
        # idxs = range(len(drugs))
        # random.shuffle(idxs)
        # drugs = drugs[idxs]
        # targets = targets[idxs]
        #
        # neg_drugs, neg_targets = np.where(W < 1)
        # idxs_neg = range(len(neg_drugs))
        # random.shuffle(idxs_neg)
        # neg_drugs = neg_drugs[idxs_neg]
        # neg_targets = neg_targets[idxs_neg]
        # idx = 0
        # for _ in xrange(np.sum(data != 0)):
        #     # <u,i>=1
        #     u = drugs[idx]
        #     i = targets[idx]
        #     v = neg_drugs[idx]
        #     j = neg_targets[idx]
        #     idx = idx + 1
        #     yield u, i, v, j

        assert len(tr_drugs) == len(tr_targets)
        idxs = range(len(tr_drugs))
        random.shuffle(idxs)
        tr_drugs = tr_drugs[idxs]
        tr_targets = tr_targets[idxs]

        train_data = data[tr_drugs, tr_targets]

        tr_pos_idx = np.where(train_data > 0)
        tr_neg_idx = np.where(train_data < 1)

        for idx in np.ravel(tr_pos_idx):
            # <u,i> should be 1 in intmat
            u = tr_drugs[idx]
            i = tr_targets[idx]
            assert data[u, i] == 1
            neg_idx = random.choice(tr_neg_idx[0])
            v = tr_drugs[neg_idx]
            j = tr_targets[neg_idx]
            assert data[v,j] == 0
            yield u,i,v,j

    def fix_model(self,W,intMat,drugMat,targetMat,seed,isverbose=False,sampling_type=0,debug=0):
        self.num_drugs, self.num_targets = intMat.shape
        # x and y are the indices of train data
        x, y = np.where(W > 0)
        self.train_drugs = set(x.tolist())
        self.train_targets = set(y.tolist())
        if seed is None:
            self.U = np.sqrt(1/float(self.K))*np.random.normal(size=(self.num_drugs, self.K))
            self.V = np.sqrt(1/float(self.K))*np.random.normal(size=(self.num_targets, self.K))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1 / float(self.K)) * prng.normal(size=(self.num_drugs, self.K))
            self.V = np.sqrt(1 / float(self.K)) * prng.normal(size=(self.num_targets, self.K))
        self.intMat = self.cfix * intMat * W
        x_1, y_1 = np.where(self.intMat > 0)
        self.train_drugs, self.train_targets = set(x_1.tolist()), set(y_1.tolist())
        self.construct_neighborhood(drugMat, targetMat)
        loss=[]
        for t in xrange(self.max_iter):
            if isverbose:
                print 'starting iteration {0}'.format(t)
            if sampling_type==0:
                for u,i,j in self.draw_train_sample(x,y,intMat):
                    if isverbose:
                        print "samples:{0}".format(debug)
                    error=self.update_factors(self.loss_type,self.neighbor_reg,u, i, j)
                    loss.append(error)
                    if isverbose:
                        debug=debug+1
            elif sampling_type==1:
                for u, i, v, j in self.draw_train_sample_special(x,y,intMat):
                    if debug:
                        print "positive:<{0},{1}>={2};negtive:<{3},{4}>={5}".format(u,i,intMat[u,i],v,j,intMat[v,j])
                    if isverbose:
                        print "samples:{0}".format(debug)
                    error = self.update_factors(self.loss_type,self.neighbor_reg,u, i, v, j)
                    loss.append(error)
            else:
                raise Exception('Wrong sampling type')
            if isverbose:
                print 'iteration {0}: loss = {1}'.format(t, np.mean(loss))

    def nb_reg(self, isDrug):
        if isDrug:
            return self.lambda_d * self.U + self.alpha * np.dot(self.DL, self.U)
        else:
            return self.lambda_t * self.V + self.beta * np.dot(self.TL, self.V)

    def update_factors(self,*train_sample):
        def update(error, vecotor_to_update, regular, deta, reg_vector, loss_type=0,):
            if loss_type==0:
                return vecotor_to_update + self.eta * (error * deta + regular * reg_vector)
            if loss_type==1:
                return vecotor_to_update + self.eta * (error * deta - regular * reg_vector)

        def remove_overflow(val):
                if val > 100:
                    val = 100
                if val < -100:
                    val = -100
                return val

        P = self.nb_reg(isDrug=True)
        Q = self.nb_reg(isDrug=False)

        def get_reg(idx,matx,isDrug):
            if self.neighbor_reg == 0:
                if isDrug:
                    reg_vector = self.U[idx, :]
                else:
                    reg_vector = self.V[idx, :]
            else:
                reg_vector = matx[idx, :]
            return  reg_vector


        num_params = len(train_sample)
        # print num_params
        if num_params == 5:
            loss_type, neighbor_reg, u, i, j = train_sample
            x_uij = np.dot(self.U[u, :], self.V[i, :].T) - np.dot(self.U[u, :], self.V[j, :].T)
            x_uij = remove_overflow(x_uij)
            # logistic function
            if loss_type==0:
                # avoid overflow
                error = (1. / (1 + np.exp(x_uij)))
            # square loss
            if loss_type==1:
                error = 1-x_uij

            # update [U]_u:
            deta = self.V[i, :] - self.V[j, :]
            reg_vector = get_reg(u,P,isDrug=True)
            # self.U[u, :] = update(error, self.U[u, :], self.lambda_u, deta, reg_vector, loss_type=loss_type)
            temp_U_u = update(error, self.U[u, :], self.lambda_u, deta, reg_vector, loss_type=loss_type)
            # update [I]_if
            deta = self.U[u, :]
            reg_vector = get_reg(i, Q, isDrug=False)
            # self.V[i, :] = update(error, self.V[i, :], self.lambda_i, deta, reg_vector, loss_type=loss_type)
            temp_V_i= update(error, self.V[i, :], self.lambda_i, deta, reg_vector, loss_type=loss_type)
            # update [I]_jf
            deta = -self.U[u, :]
            reg_vector = get_reg(j, Q, isDrug=False)
            # self.V[j, :] = update(error, self.V[j, :], self.lambda_j, deta, reg_vector, loss_type=loss_type)
            temp_V_j = update(error, self.V[j, :], self.lambda_j, deta, reg_vector, loss_type=loss_type)

            # update
            self.U[u, :] = temp_U_u
            self.V[i, :] = temp_V_i
            self.V[j, :] = temp_V_j

        elif num_params==6:
                loss_type, neighbor_reg, u,i,v,j = train_sample
                x_uivj = np.dot(self.U[u, :], self.V[i, :].T) - np.dot(self.U[v, :], self.V[j, :].T)
                # avoid overflow
                x_uivj = remove_overflow(x_uivj)
                # logistic function
                if loss_type == 0:
                    # avoid overflow
                    error = (1. / (1 + np.exp(x_uivj)))
                # square loss
                if loss_type == 1:
                    error = 1 - x_uivj
                # update [U]_uf
                deta = self.V[i, :]
                reg_vector = get_reg(u, P, isDrug=True)
                # self.U[u, :] = update(error, self.U[u, :], self.lambda_u, deta, reg_vector, loss_type=loss_type)
                temp_U_u = update(error, self.U[u, :], self.lambda_u, deta, reg_vector, loss_type=loss_type)
                # updata [U]_vf
                deta = -self.V[j,:]
                reg_vector = get_reg(u, P, isDrug=True)
                # self.U[v, :] = update(error, self.U[v, :], self.lambda_u, deta, reg_vector, loss_type=loss_type)
                temp_U_v = update(error, self.U[v, :], self.lambda_u, deta, reg_vector, loss_type=loss_type)
                # update [I]_if
                deta = self.U[u, :]
                reg_vector = get_reg(u, P, isDrug=True)
                # self.V[i, :] = update(error, self.V[i, :], self.lambda_i, deta, reg_vector, loss_type=loss_type)
                temp_V_i = update(error, self.V[i, :], self.lambda_i, deta, reg_vector, loss_type=loss_type)
                # update [I]_jf
                deta = -self.U[u, :]
                reg_vector = get_reg(u, P, isDrug=True)
                # self.V[j, :] = update(error, self.V[j, :], self.lambda_j, deta, reg_vector, loss_type=loss_type)
                temp_V_j = update(error, self.V[j, :], self.lambda_j, deta, reg_vector, loss_type=loss_type)

                # update
                self.U[u, :] = temp_U_u
                self.U[v, :] = temp_U_v
                self.V[i, :] = temp_V_i
                self.V[j, :] = temp_V_j

        else:
            raise Exception('wrong number of args')
        return error

    def evaluation(self,test_data,test_label):
        if self.neighbor_reg:
            dinx = np.array(list(self.train_drugs))
            DS = self.dsMat[:, dinx]
            tinx = np.array(list(self.train_targets))
            TS = self.tsMat[:, tinx]
            scores = []
            for d,t in test_data:
                if d in self.train_drugs:
                    if t in self.train_targets:
                        val = np.dot(self.U[d, :],self.V[t, :])
                    else:
                        jj = np.argsort(TS[t, :])[::-1][:self.K2]
                        val = np.sum(self.U[d, :] * np.dot(TS[t, jj], self.V[tinx[jj], :])) / np.sum(TS[t, jj])
                else:
                    if t in self.train_targets:
                        ii = np.argsort(DS[d, :])[::-1][:self.K2]
                        val = np.sum(np.dot(DS[d, ii], self.U[dinx[ii], :]) * self.V[t, :]) / np.sum(DS[d, ii])
                    else:
                        ii = np.argsort(DS[d, :])[::-1][:self.K2]
                        jj = np.argsort(TS[t, :])[::-1][:self.K2]
                        v1 = DS[d, ii].dot(self.U[dinx[ii], :]) / np.sum(DS[d, ii])
                        v2 = TS[t, jj].dot(self.V[tinx[jj], :]) / np.sum(TS[t, jj])
                        val = np.sum(v1 * v2)
                scores.append(np.exp(val) / (1 + np.exp(val)))
        else:
            uu, ii = test_data[:, 0], test_data[:, 1]
            scores = np.sum(self.U[uu, :] * self.V[ii, :], axis=1)

        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

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



def train(model, cv_data, intMat, drugMat, targetMat):
    aupr, auc = [], []
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val = model.evaluation(test_data, test_label)
            aupr.append(aupr_val)
            auc.append(auc_val)
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64)

if __name__=="__main__":
    import time
    from functions import *
    folds=5
    latent_k=100
    # max iters
    max_iter = 100
    debug=0
    cvs=3
    #0: classical sample, uij; 1: ui,vj
    sampling_type=0
    #0: logistic loss; 1: square loss
    loss_type = 1
    #0: do not use neighbor regularization
    neighbor = 1

    print "params: sampling_type={0}, max_iter={1}, loss_type={2}, neighbor={3}, CVS={4}".format(sampling_type,max_iter,
                                                                                                 loss_type,neighbor,
                                                                                                 cvs)
    seeds = [7771, 8367, 22, 1812, 4659]
    for dataset in ["nr", "gpcr", "ic", "e"]:
        print "**********training dataset:{0}*************".format(dataset)

        intMat, drugMat, targetMat = load_data_from_file(dataset, "./data/")
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
            fold_count=0
            for W, test_data, test_label in cv_data[seed]:
                # print "Fold:{0}".format(fold_count)
                fold_count=fold_count+1
                model=BPR(max_iter=100, K=latent_k, loss_type=loss_type, neighbor_reg=neighbor)
                model.fix_model(W,X,D,T,seed,sampling_type=sampling_type)
                aupr_val, auc_val = model.evaluation(test_data, test_label)
                aupr.append(aupr_val)
                auc_list.append(auc_val)
            if debug:
                break
        aupr_vec=np.array(aupr, dtype=np.float64)
        auc_vec=np.array(auc_list, dtype=np.float64)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print "auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)
        if debug:
            break
