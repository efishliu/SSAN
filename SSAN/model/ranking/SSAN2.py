from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix
import numpy as np
from util.loss import bpr_loss
import os
from util import config
from math import sqrt
from util.io import FileIO
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Recommended Maximum Epoch Setting: LastFM 120 Douban 30 Yelp 30
# A slight performance drop is observed when we transplanted the model from python2 to python3. The cause is unclear.

class SSAN2(SocialRecommender,GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readConfiguration(self):
        super(SSAN2, self).readConfiguration()
        args = config.OptionConf(self.config['SSAN2'])
        self.n_layers = int(args['-n_layer'])
        self.ss_rate1 = float(args['-ss_rate1'])
        self.ss_rate2 = float(args['-ss_rate2'])
        self.r_pos = 3.0

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix

    def buildStatusMatrix(self):
        S = self.buildSparseRelationMatrix()
        B = S.multiply(S.T)
        U = S - B
        U = U.todense()
        return U

    def buildSparseRatingMatrix(self):
        #print('='*80)
        #print("Building Ratings Matrix\n")
        row, col, entries = [], [], []
        i = 0
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
            i += 1
        print('user-item rating count:',str(i),'\n')
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMatrix
    
    def buildSparsePolarityRatingMatrix(self):
        r_pos = self.r_pos
        row_p, col_p, entries_p = [], [], []
        row_n, col_n, entries_n = [], [], []
        row_r, col_r, entries_r = [], [], []
        i,j = 0,0
        #print('='*80)
        #print("Building Polarity Ratings Matrix\n")
        allRatingsData = FileIO.loadDataSet(self.config, self.config['ratings'],binarized = False,threshold=1)
        allRatingsDict = {}
        for iter in allRatingsData:
            userid = iter[0]    #原始user id
            itemid = iter[1]
            rating = iter[2]
            allRatingsDict[str(userid)+'_'+str(itemid)] = rating
        for pair in self.data.trainingData:
            # symmetric matrix
            userid = self.data.user[pair[0]]    #转换后的user id 
            itemid = self.data.item[pair[1]]
            rating = allRatingsDict[str(pair[0])+'_'+str(pair[1])]
            if float(rating) > r_pos:
                row_p += [userid]
                col_p += [itemid]
                entries_p += [1.0]
                row_r += [userid]
                col_r += [itemid]
                entries_r += [1.0]
                i += 1
            else:
                row_n += [userid]
                col_n += [itemid]
                entries_n += [1.0]
                row_r += [userid]
                col_r += [itemid]
                entries_r += [-1.0]
                j += 1
            #print('userid:%s,itemid:%s,rating:%s' % (str(self.data.user[pair[0]]),str(self.data.item[pair[1]]),str(pair[2])))
        #print("positive user-item rating count:",str(i),'\n')
        #print("negtive  user-item rating count:",str(j),'\n')
        #print("user-item rating count:",str(i+j),'\n')
        #print('='*80)
        rating_p_Matrix = coo_matrix((entries_p, (row_p, col_p)), shape=(self.num_users,self.num_items),dtype=np.float32)
        rating_n_Matrix = coo_matrix((entries_n, (row_n, col_n)), shape=(self.num_users,self.num_items),dtype=np.float32)
        rating_Matrix = coo_matrix((entries_r, (row_r, col_r)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return rating_p_Matrix,rating_n_Matrix,rating_Matrix

    def buildJointAdjacency(self):
        indices = [[self.data.user[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData]
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_users, self.num_items])
        return norm_adj

    def buildMotifInducedAdjacencyMatrix(self):
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        Y_P,Y_N,Y_R = self.buildSparsePolarityRatingMatrix()    #R+,R-,R+-
        self.userAdjacency = Y.tocsr()
        self.itemAdjacency = Y.T.tocsr()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        #A1 = C1
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        #A2 = C2
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        #A3 = C3
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        #A5 = C5
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        #A8_ = (Y_P.dot(Y_P.T)).multiply(B) + (Y_N.dot(Y_N.T)).multiply(B)
        #A9_ = (Y_P.dot(Y_P.T)).multiply(U) + (Y_N.dot(Y_N.T)).multiply(U)
        #A8 = (Y.dot(Y.T)).multiply(B)
        #A9 = (Y.dot(Y.T)).multiply(U)
        A8 = (Y_R.dot(Y_R.T)).multiply(B)
        A9 = (Y_R.dot(Y_R.T)).multiply(U)
        A9 = A9+A9.T
        #A10  = Y.dot(Y.T)-A8-A9
        
        #A10 = Y.dot(Y.T)+Y_N.dot(Y_N.T)-A8-A9
        A12 = Y_P.dot(Y_N.T) + Y_N.dot(Y_P.T)

        A10  = Y_P.dot(Y_P.T)+Y_N.dot(Y_N.T)-A8-A9-A12
        #A10  = (Y_R.dot(Y_R.T)) -A8-A9

        #addition and row-normalization
        H_s = sum([A1,A2,A3,A4,A5,A6,A7])
        print("="*80)
        print("H_s count:",str(H_s.count_nonzero()))
        #H_s = S
        H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1))
        
        H_j = sum([A8,A9])
        H_j = H_j.multiply(H_j>1)
        print("H_j count:",str(H_j.count_nonzero()))
        H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
        
        H_p = A10
        H_p = H_p.multiply(H_p>1)
        print("H_p count:",str(H_p.count_nonzero()))
        H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))
        

        #add negtive friends
        H_n = A12
        print("H_n count:",str(H_n.count_nonzero()))
        H_n = H_n.multiply(1.0/H_n.sum(axis=1).reshape(-1, 1))
        
        print("="*80)

        return [H_s,H_j,H_p,H_n]

    def adj_to_sparse_tensor(self,adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj

    def strust_self_supervision(self,em,h_s,h_j,h_p):
        def score(x1,x2):
            return tf.reduce_sum(tf.multiply(x1,x2),1)
        def struct_supervised_gating(em):
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em, self.weights['strust_wight'])+self.weights['strust_bias']))
        user_embeddings = struct_supervised_gating(em)
        # user_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        social_neig_embeddings = tf.sparse_tensor_dense_matmul(h_s,user_embeddings)
        positive_neig_embeddings = tf.sparse_tensor_dense_matmul(h_p,user_embeddings)
        joint_neig_embeddings = tf.sparse_tensor_dense_matmul(h_j,user_embeddings)

        pos_sim = score(user_embeddings,positive_neig_embeddings)
        social_sim = score(user_embeddings,social_neig_embeddings)
        joint_sim = score(user_embeddings,joint_neig_embeddings)
        #struct_loss = tf.reduce_sum(-tf.log(tf.sigmoid(joint_sim-pos_sim))-tf.log(tf.sigmoid(pos_sim-social_sim)))
        struct_loss = tf.reduce_sum(-tf.log(tf.sigmoid(joint_sim-social_sim))-tf.log(tf.sigmoid(social_sim-pos_sim)))
        return struct_loss
     

    def status_self_supervision(self,em,status_matric):
        user_embeddings = tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating1'])+self.weights['gating_bias1']))
        user_status_score = tf.nn.tanh(tf.matmul(user_embeddings,self.weights['status_wight'])+self.weights['status_bias'])
        user_status_diff = -(user_status_score - tf.transpose(user_status_score)) #v-u
        user_status_diff = tf.multiply(user_status_diff,status_matric>0)
        status_loss = tf.reduce_sum(tf.square(status_matric-user_status_diff))
        return status_loss


    
    def initModel(self):
        super(SSAN2, self).initModel()
        print("no self-supervised sign")
        print("without s channel")
        M_matrices = self.buildMotifInducedAdjacencyMatrix()
        #status_matric = self.buildStatusMatrix()
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.n_channel = 3
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        #define learnable paramters
        for i in range(self.n_channel):
            self.weights['gating%d' % (i+1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' %(i+1)] = tf.Variable(initializer([1, self.emb_size]), name='g_W_b_%d_1' % (i + 1))
            #self.weights['sgating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='sg_W_%d_1' % (i + 1))
            #self.weights['sgating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]), name='sg_W_b_%d_1' % (i + 1))
        self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at')
        self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm')
        self.weights['final_attention'] = tf.Variable(initializer([1, self.emb_size]), name='final_at')
        self.weights['final_attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='final_atm')
        #self.weights['status_wight'] = tf.Variable(initializer([self.emb_size,1]), name='status_w')
        #self.weights['status_bias'] = tf.Variable(initializer([1,]), name='status_b')
        #self.weights['strust_wight'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='strust_w')
        #self.weights['strust_bias'] = tf.Variable(initializer([1, self.emb_size]), name='strust_b')
        #define inline functions
        def self_gating(em,channel):
            return tf.multiply(em,tf.nn.sigmoid(tf.matmul(em,self.weights['gating%d' % channel])+self.weights['gating_bias%d' %channel]))
        
        def channel_attention(*channel_embeddings):
            weights = []
            for embedding in channel_embeddings:
                weights.append(tf.reduce_sum(tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])),1))
            score = tf.nn.softmax(tf.transpose(weights))
            mixed_embeddings = 0
            for i in range(len(weights)):
                mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
            return mixed_embeddings,score

        def final_attention(*embeddings):
            weights = []
            for embedding in embeddings:
                weights.append(tf.reduce_sum(tf.multiply(self.weights['final_attention'], tf.matmul(embedding, self.weights['final_attention_mat'])),1))
            score = tf.nn.softmax(tf.transpose(weights))
            mixed_embeddings = 0
            for i in range(len(weights)):
                mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(embeddings[i])))
            return mixed_embeddings,score
        #initialize adjacency matrices
        #H_s = M_matrices[0]
        #H_s = self.adj_to_sparse_tensor(H_s)
        H_j = M_matrices[1]
        H_j = self.adj_to_sparse_tensor(H_j)
        H_p = M_matrices[2]
        H_p = self.adj_to_sparse_tensor(H_p)
        H_n = M_matrices[3]
        H_n = self.adj_to_sparse_tensor(H_n)
        R = self.buildJointAdjacency()
        #self-gating
        #user_embeddings_c1 = self_gating(self.user_embeddings,1)
        user_embeddings_c2 = self_gating(self.user_embeddings, 1)
        user_embeddings_c3 = self_gating(self.user_embeddings, 2)
        simple_user_embeddings = self_gating(self.user_embeddings,3)
        #all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        item_embeddings = self.item_embeddings
        all_embeddings_i = [item_embeddings]


        #multi-channel convolution
        for k in range(self.n_layers):
            mixed_embedding = channel_attention(user_embeddings_c2, user_embeddings_c3)[0] + simple_user_embeddings / 2
            #mixed_embedding = channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)[0] + simple_user_embeddings / 2
            #Channel S
            #user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_s,user_embeddings_c1)
            #norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1)
            #all_embeddings_c1 += [norm_embeddings]
            #Channel J
            user_embeddings_c2 = tf.sparse_tensor_dense_matmul(H_j, user_embeddings_c2)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c2, axis=1)
            all_embeddings_c2 += [norm_embeddings]
            #Channel P
            '''
            if k == 0:
                A_pos = H_p
                A_neg = H_n
            else:
                A_pos_ = A_pos.dot(A_pos) + A_neg.dot(A_neg)
                A_neg_ = A_pos.dot(A_neg) + A_neg.dot(A_pos)
                A_pos += A_pos_
                A_neg += A_neg_
            '''
            user_embeddings_c3 = tf.sparse_tensor_dense_matmul(H_p, user_embeddings_c3)
            #user_embeddings_c3 = tf.sparse_tensor_dense_matmul(self.adj_to_sparse_tensor(A_pos), user_embeddings_c3)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c3, axis=1)
            all_embeddings_c3 += [norm_embeddings]
            # item convolution
            new_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding)
            norm_embeddings = tf.math.l2_normalize(new_item_embeddings, axis=1)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = tf.sparse_tensor_dense_matmul(R, item_embeddings)
            all_embeddings_simple += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
            item_embeddings = new_item_embeddings
        #averaging the channel-specific embeddings
        #user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        user_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
        simple_user_embeddings = tf.reduce_sum(all_embeddings_simple, axis=0)
        item_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)
        #aggregating channel-specific embeddings
        self.final_item_embeddings = item_embeddings
        #self.final_user_embeddings,self.attention_score = channel_attention(user_embeddings_c1,user_embeddings_c2,user_embeddings_c3)
        #final_user_embeddings,self.attention_score = channel_attention(user_embeddings_c1,user_embeddings_c2,user_embeddings_c3)
        final_user_embeddings,self.attention_score = channel_attention(user_embeddings_c2,user_embeddings_c3)
        #self.final_user_embeddings += simple_user_embeddings/2
        self.final_user_embeddings,final_score = final_attention(final_user_embeddings,simple_user_embeddings)

        #create self-supervised loss
        #H_p = self.adj_to_sparse_tensor(H_p)
        #H_n = self.adj_to_sparse_tensor(H_n)
        self.ss_loss1,self.ss_loss2 = 0,0
        #self.ss_loss1 = self.strust_self_supervision(self.final_user_embeddings,H_s,H_j,H_p)
        #self.ss_loss2 = self.balance_self_supervision(self.final_user_embeddings,H_p,H_n)
        #self.ss_loss2 = self.status_self_supervision(self.final_user_embeddings,status_matric)
        #embedding look-up
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)



    def buildModel(self):
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001*tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        total_loss = rec_loss+reg_loss 
        opt = tf.train.AdamOptimizer(self.lRate)
        train_op = opt.minimize(total_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # Suggested Maximum epoch Setting: LastFM 120 Douban 30 Yelp 30
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l1 = self.sess.run([train_op, rec_loss],
                                     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print(self.foldInfo,'training:', epoch + 1, 'batch', n, 'rec loss:', l1)#,'ss_loss',l2
            self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])
            self.ranking_performance(epoch)
    #self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
        self.U,self.V = self.bestU,self.bestV

    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items