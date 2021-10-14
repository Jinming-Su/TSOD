import numpy as np
import os

import torch 
from PIL import Image 
from torch.autograd import Variable 
from torchvision import transforms 
import sys 
sys.path.append('../') 
from config import cityscapes_val_path, cityscapes_train_path
#dutomron_path, hkuis_path, dutste_path, xpie_path

from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from fpn_dual import FPN_dual

torch.manual_seed(2019)

# set which gpu to use
torch.cuda.set_device(0)

ckpt_path = './ckpt'

fw = open(ckpt_path + '/evaluation.txt', 'a+')

args = {
    'resize': [256, 512],
    'max_iter': 6000,
    'save_interval': 3000,  # your snapshot filename (exclude extension name)
    'save_results': True  # whether to save the resulting masks
}

img_transform = transforms.Compose([
    
    transforms.Resize(args['resize']),
    transforms.ToTensor(),
    transforms.Normalize([0.2869, 0.3251, 0.2839], [0.1870, 0.1902, 0.1872])
])
to_pil = transforms.ToPILImage()

#to_test = {'cityscapes_train': cityscapes_train_path}
to_test = {'cityscapes_val': cityscapes_val_path}
#to_test = {'cityscapes_val': cityscapes_val_path, 'hkuis': hkuis_path, 'pascal': pascals_path, 'sod': sod_path, 'dutomron': dutomron_path}
import time

def main():
    net = FPN_dual().cuda()
    
    for model_i in range(180000,181500,5000):
        print ('starting iter %d for testing\n' % model_i)
        net.load_state_dict(torch.load(os.path.join(ckpt_path, str(model_i) + '.pth')))
        net.eval()

        results = {}

        with torch.no_grad():

            for name, root in to_test.items():

                precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
                mae_record = AvgMeter()

                save_path = os.path.join(ckpt_path, name + '/' + str(model_i))
                if args['save_results']:
                    check_mkdir(save_path)
                    check_mkdir(save_path + '_b')
                    check_mkdir(save_path + '_i')

                img_list = [os.path.splitext(f)[0] for f in os.listdir(root+'/image') if f.endswith('.png')]

                idx_i = 0
                import time
                start = time.time()
                for idx, img_name in enumerate(img_list):
                    idx_i += 1
                    if idx_i % 100 == 0:
                        print('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))

                    #start = time.time()
                    img = Image.open(os.path.join(root + '/image', img_name + '.png')).convert('RGB')
                    img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
                    prediction = net(img_var)
                    prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))




                    #gt = np.array(Image.open(os.path.join(root + '/groundtruth', img_name + '.png')).convert('L'))
                    #precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                    #for pidx, pdata in enumerate(zip(precision, recall)):
                    #    p, r = pdata
                    #    precision_record[pidx].update(p)
                    #    recall_record[pidx].update(r)
                    #mae_record.update(mae)

                    if args['save_results']:
                        import cv2
                        prediction = cv2.resize(prediction, (np.array(img).shape[1], np.array(img).shape[0]))
                        Image.fromarray(prediction).save(os.path.join(save_path + '/' + img_name + '.png'))
                    #print(time.time() - start)

                #fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                #                        [rrecord.avg for rrecord in recall_record])

                #results[name] = {'mae': mae_record.avg, 'fmeasure': fmeasure, 'iter': model_i}

                #fw.write(str(results) + '\n')
                #print( results)
                end = time.time() 
    print((end -start )*1.0/500)
    fw.close()  

if __name__ == '__main__':
    main()
