from label_defeat import label_object, myThread, show_classes, show_abnormal_state
from utils import SortFiles, RenameFiles
import os, time
import argparse

if __name__ == "__main__":
    item = 'hxqyfywyc'
    weathers = ['weather0']# ['weather0', 'weather1', 'weather2', 'weather3', 'weather4', 'weather5', 'weather6', 'weather7', 'weather8']
    print(weathers)
    for weather in weathers:
        print('now is sorting at {} {}'.format(item, weather))
        source_path = os.path.join(r'D:\Documents\AirSim', item, weather)
        target_path = source_path.replace(item, item + '_sorted')

        # sort stage
        SortFiles(source_path, target_path)
        RenameFiles(target_path+'\\ori', '.jpg')
        RenameFiles(target_path+'\\seg', '.jpg')
        
        print('{} anno start!'.format(item))

        # anno stage
        parser = argparse.ArgumentParser()
        parser.add_argument('--seg_path', type=str, default = os.path.join(target_path, 'seg'), help='The path saved the segmentation img')
        parser.add_argument('--ori_path', type=str, default = os.path.join(target_path, 'ori'), help='The path saved the origin img')
        parser.add_argument('--label_path', type=str, default = os.path.join(target_path, 'anno'), help='The path saved the label xml')
        # parser.add_argument('--reverse', type=bool, default=False)

        config = parser.parse_args()

        thread1 = myThread(1, 'Thread1', config)
        thread2 = myThread(2, 'Thread2', config)

        thread1.start()
        time.sleep(2)
        thread2.start()
        thread1.join()
        thread2.join()

        usable_file_list = show_classes(config.label_path)
        print(len(usable_file_list))
        show_abnormal_state(config)