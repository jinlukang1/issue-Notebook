import argparse
import os, time, random
from label_object import myThread
from utils import SortFiles, RenameFiles, wash_data, gen_each_train_test

if __name__ == '__main__':
    source_path = r'd:\Documents\AirSim\biaoji_D\weather8'
    target_path = r'd:\Documents\AirSim\biaoji_D\sorted\weather8'
    # #
    SortFiles(source_path, target_path)
    RenameFiles(os.path.join(target_path, 'seg'), '.jpg')
    RenameFiles(os.path.join(target_path, 'ori'), '.jpg')
    wash_data(os.path.join(target_path, 'seg'))
    RenameFiles(os.path.join(target_path, 'seg'), '.jpg')
    RenameFiles(os.path.join(target_path, 'ori'), '.jpg')

    random.seed(55)
    gen_each_train_test(target_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', type=str, default=os.path.join(target_path, 'seg'), help='The path saved the segmentation img')
    parser.add_argument('--ori_path', type=str, default=os.path.join(target_path, 'ori'), help='The path saved the origin img')
    parser.add_argument('--label_path', type=str, default=os.path.join(target_path, 'anno'), help='The path saved the label xml')
    # parser.add_argument('--reverse', type=bool, default=False)

    config = parser.parse_args()

    thread1 = myThread(1, 'Thread1', config)
    thread2 = myThread(2, 'Thread2', config)

    thread1.start()
    time.sleep(2)
    thread2.start()
    thread1.join()
    thread2.join()



