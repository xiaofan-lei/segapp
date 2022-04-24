from flask import render_template, request

from urllib.request import urlopen
import numpy as np
import cv2
import re
import os
from skimage import color

#get the registred segmentation model
from app import app
from app.azure_connect import PSPNetInferrer, mask_to_labelIds, mask_plot, upload_file
model = PSPNetInferrer()
IMG_URL=app.config['IMG_URL']

@app.route('/')
def main():
    #images from url
    return render_template("index.html", IMG_URL=IMG_URL, imglist=app.config['IMG_LIST'])

@app.route('/segmentation', methods=['GET','POST'])
def segmentation():
    # image serialization
    img_file = request.form.get('comp_select')
    url = IMG_URL + img_file
    resp = urlopen(url)
    arr = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(arr, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #generate a prediction mask
    pred = model.infer(image)
    mask = mask_to_labelIds(pred)

    #add the mask to image
    result_image = color.label2rgb(mask, image)
    result_image = (result_image * 255).astype(np.uint8)

    # save prediction files
    mask_file = re.sub("leftImg8bit.png", "labelIds.png", img_file)
    masked_img_file = re.sub("leftImg8bit.png", "leftImg8bit_labelIds.png", img_file)

    cv2.imwrite('./app/static/upload/' + masked_img_file, result_image)
    cv2.imwrite('./app/static/upload/' + mask_file, mask)
    #upload to the dataset
    mask_file_path = os.path.join(app.root_path, 'static/upload', mask_file)
    TARGET_PATH='UI/04-15-2022_073440_UTC/citydata/gtFine/test'
    upload_file(mask_file_path, TARGET_PATH)

    return render_template('segmentation.html',
                           IMG_URL=IMG_URL,
                           TARGET_PATH=TARGET_PATH,
                           mask_url=app.config['DATASET_URL'] + TARGET_PATH + '/' + mask_file,
                           img_file=img_file,
                           plot_mask=mask_plot(mask),
                           mask_file=mask_file,
                           masked_img_file=masked_img_file
                           )

# To get one variable, tape app.config['MY_VARIABLE']
if __name__ == "__main__":
        app.run()