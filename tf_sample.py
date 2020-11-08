import tensorflow as tf
import numpy as np
import random

GEN_RANK_NUM = 4+1
LENGTH = 40
batch_size = 128
K = 20


def tf_sample_and_test(prob_table, temperature, LENGTH=70, GEN_RANK_NUM=5, k=K):
    prob_table = tf.convert_to_tensor(prob_table)
    if len(prob_table.shape) == 3:
        prob_table = prob_table[0:,0:,:-1]
    elif len(prob_table.shape) == 2:
        prob_table = np.reshape(prob_table, (-1, LENGTH, GEN_RANK_NUM))[:,:,0:-1]   

    def set_value(matrix, x, y, col_len=4,val=0.):
        # 提取出要更新的行
        row = tf.gather(matrix, x)
        # 构造这行的新数据
        new_row = tf.concat([row[:y], [val]*(col_len-y)], axis=0)
        # 使用 tf.scatter_update 方法进正行替换
        return tf.scatter_update(matrix, x, new_row)  # matrix必须是var

    def tf_sample_one(table):
        random_shifts = np.random.random([LENGTH,GEN_RANK_NUM-1])  # [70,4]
        random_shifts /= random_shifts.sum(axis = 0)[np.newaxis,:]
        result = []
    
        for m in range(GEN_RANK_NUM-1):  # 4个位置
            table_m = table[:,m]  # [70,1]
    
            table_m /= tf.reduce_sum(table_m)  # 这里要控制执行顺序，先计算这里，再置零
            shifted_probabilities = random_shifts[:,m] - table_m
            with tf.control_dependencies([table_m,shifted_probabilities]):
    
                l = tf.argmin(shifted_probabilities)  # 找出最小的index
                table = set_value(table,x=l,y=m,col_len=GEN_RANK_NUM-1,val=0.)
                result.append(l)
        return tf.reshape(result,[1,len(result)])

    def tf_check_repeat(checked, cand_set):
        flag = tf.convert_to_tensor(False)
        
        for cand_tensor in cand_set:
            eq = tf.equal(checked,cand_tensor)
            cond = tf.equal(tf.reduce_sum(tf.cast(eq,tf.int32)),GEN_RANK_NUM-1)
           
            flag = tf.cond(cond, lambda: tf.cast(True, tf.bool), lambda: flag)
        return flag  # 但凡有一个重复则true
        

    prob_table = tf.math.exp(prob_table * temperature)  # [bs, LENGTH, GEN_RANK_NUM-1]  [3,70,4]
    
    # 从每条候选概率[70,4]中抽样20个即[20,4]
    i = tf.constant(0,name='cons_i')
    prob_table_frame = tf.Variable(prob_table[0],name='prob_frame')
    sample_res = tf.Variable([],name='sample_res',dtype=tf.int64)

    def condition(i, prob_table,_):
        return i<tf.shape(prob_table)[0]
    
    def loop_body(i, prob_table,res):

        sample_set = []
        num = k
        cnt = 0
        while num > 0:  # 抽取20个sku的位置概率,K=2
            tf.add_to_collection('i', value=[i])
            cnt += 1
            print('num:',num)
            print('count:',cnt)

            table = tf.assign(prob_table_frame, prob_table[i])
            print('table:',table)
            tmp = tf_sample_one(table)  # [70,4] -> [1,4]
            if len(sample_set)==0:
                sample_set_concat = tmp
                sample_set.append(tmp)
                num -= 1
            else:
                tmp_len = len(sample_set)#sample_set_concat.get_shape().as_list()[0]
                sample_set_concat = tf.cond(tf_check_repeat(tmp,sample_set), 
                        true_fn=lambda: sample_set_concat, 
                        false_fn=lambda: tf.concat([sample_set_concat,tmp],0))  
                if sample_set_concat.get_shape().as_list()[0]!=tmp_len:
                    sample_set.append(tmp)
                    num -= 1

            if sample_set_concat.get_shape().as_list()[0] == k:
                break
        tmpp = tf.reshape(sample_set_concat,[k*(GEN_RANK_NUM-1),])
        res = tf.concat([res, tmpp], 0)

        i += 1
        return [i,prob_table,res]
        
    _i, _prob_table,res = tf.while_loop(condition, 
                                        loop_body, 
                                        [i, prob_table, sample_res],
                                        shape_invariants=[i.get_shape(),
                                                          prob_table.get_shape(),
                                                          tf.TensorShape([None])])
    return res  # [bs, K, GEN_RANK_NUM-1]

tf.reset_default_graph()
np.random.seed(1)
a = np.random.random_sample((batch_size, LENGTH,GEN_RANK_NUM)).astype(np.float32)
#a = np.reshape(a,[3,70*5])
tensor_a = tf.convert_to_tensor(a)
a[0]


inputs = tf.placeholder(tf.float32,shape=(None,LENGTH,GEN_RANK_NUM))

tf_k_list = tf_sample_and_test(
    #a,
    inputs, 
    0.4,  LENGTH=LENGTH, GEN_RANK_NUM=GEN_RANK_NUM, k=K)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

k_list = sess.run(k_list,feed_dict={inputs:a})
#k_list=sess.run([tf.reshape(tf_k_list,[-1,K,GEN_RANK_NUM-1])])
print(k_list)
