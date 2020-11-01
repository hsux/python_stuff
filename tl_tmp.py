# coding:utf-8

import tensorflow as tf
import numpy as np
import random

GEN_RANK_NUM = 4+1
LENGTH = 7
batch_size = 3
K = 2

np.random.seed(1)
a = np.random.random_sample((batch_size, LENGTH,GEN_RANK_NUM))
#a = np.reshape(a,[3,70*5])
tensor_a = tf.convert_to_tensor(a)

check = []
check_tb = []

def set_value(matrix, x, y, col_len=4,val=0.):
    row = tf.gather(matrix, x)

    new_row = tf.concat([row[:y], [val]*(col_len-y)], axis=0)

    return tf.scatter_update(matrix, x, new_row)  # matrix
	
def tf_check_repeat(checked, cand_set):
    flag = tf.convert_to_tensor(False)
    for cand_tensor in cand_set:
        flag = tf.cond(tf.cast(checked==cand_tensor, tf.bool), lambda: tf.cast(True, tf.bool), lambda: flag)
    return flag  

def tf_sample_one(table):
    #np.random.seed(0)
    random_shifts = np.random.random([LENGTH,GEN_RANK_NUM-1])  # [70,4]
    #print('------------\n',random_shifts,'-----------\n')
    random_shifts /= random_shifts.sum(axis = 0)[np.newaxis,:]
    result = []
    print('sample_one,table_shape',table)  # [7,4]
    for m in range(GEN_RANK_NUM-1):  # 
        table_m = table[:,m]  # [70,1]
        check.append((tf.reduce_sum(table_m),table_m))
        table_m /= tf.reduce_sum(table_m)  # 
        
        with tf.control_dependencies([table_m]):
            shifted_probabilities = random_shifts[:,m] - table_m
            with tf.control_dependencies([shifted_probabilities]):
                check.append((table_m,shifted_probabilities))
                l = tf.argmin(shifted_probabilities)  # 
                table = set_value(table,x=l,y=m,col_len=GEN_RANK_NUM-1,val=0.)
                result.append(l)
            
    check_tb.append(table)
    return tf.reshape(result,[1,len(result)])
	
def tf_sample_and_test(prob_table, temperature, k=K):
    prob_table = tf.convert_to_tensor(prob_table)
    if len(prob_table.shape) == 3:
        prob_table = prob_table[0:,0:,:-1]
    elif len(prob_table.shape) == 2:
        prob_table = np.reshape(prob_table, (-1, LENGTH, GEN_RANK_NUM))[:,:,0:-1]   
    res = []
    prob_table = tf.math.exp(prob_table * temperature)  # [bs, LENGTH, GEN_RANK_NUM-1]  [3,70,4]
    
   
    for i in range(prob_table.get_shape().as_list()[0]):  # bs
        print('exam:',i)
        sample_set = []  
        print('k={}'.format(k))
        num = k
        cnt = 0
        while num > 0:  # 
            cnt += 1
            print('count:',cnt)
            table  = tf.Variable(prob_table[i])  # 
            #check_tb.append(table)
            print('table:',table)
            tmp = tf_sample_one(table)  # [70,4] -> [1,4]
            if len(sample_set)==0:
                sample_set_concat = tmp
                sample_set.append(tmp)
                num -= 1
                print('first sample-set list-------------------------------')
            else:
                tmp_len = sample_set_concat.get_shape().as_list()[0]
                tf.cond(tf_check_repeat(tmp,sample_set), 
                        true_fn=lambda: sample_set_concat, 
                        false_fn=lambda: tf.concat([sample_set_concat,tmp],0))  
                if sample_set_concat.get_shape().as_list()[0]!=tmp_len:
                    sample_set.append(tmp)
                    num -= 1
                    print('second  list in stack==============================')
            if cnt > 10:
                break
        res.append(sample_set_concat)

    res = tf.concat(res,0)
    return res  # [bs, K, GEN_RANK_NUM-1]
	
tf_k_list = tf_sample_and_test(a, 0.4)
print(tf_k_list)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

k_list,ck = sess.run([tf_k_list,check])

print(k_list)

print(ck[0])