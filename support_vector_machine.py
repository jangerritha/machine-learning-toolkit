import libsvm.python.svm as svm
import libsvm.python.svmutil as svm_util
import libsvm.python.commonutil as common_util


def execute_svm():
    if __name__ == '__main__':
        input = common_util.svm_read_problem('./libsvm/heart_scale')
        #print(str(len(input)))
        model = svm_util.svm_train(input[0], input[1], '-c 1 -g 0.07')
        results = svm_util.svm_predict(input[0], input[1], model)
        #print(results)



if __name__ == '__main__':
    execute_svm()