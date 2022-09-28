import matplotlib.pyplot as plt
import torch
import flask
from flask import Flask, render_template, request
import json
import os
import numpy as np
import cv2
import pytesseract
from mrz.checker.td3 import TD3CodeChecker, get_country
import base64
import numpy as np



app = Flask(__name__)
app.env = "development"

model = torch.hub.load('./yolov5', 'custom', path='./model/best.pt', source='local')

def Crop_Image(image, data):

    ymin = int(data[1])
    ymax = int(data[3])
    xmin = int(data[0])
    xmax = int(data[2])

    crop_region = image[ymin:ymax, xmin:xmax]

    return crop_region

def Extract_Character(crop_mrz):
    crop_mrz = crop_mrz.astype('uint8')
    gray = cv2.cvtColor(crop_mrz, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)

    results = {}
    results["gray_result"] = pytesseract.image_to_string(gray, lang="ocrb")
    results["threshed_result"] = pytesseract.image_to_string(threshed, lang="ocrb")

    global check
    global fields

    for i in results:
        # print("ilk : \n"+results[i])
        ocr_result = results[i]
        P_index = ocr_result.index("P")
        ocr_result = ocr_result[P_index:]
        ocr_result = ocr_result.replace(" ", "")
        ocr_result = ocr_result.replace("$", "S")
        ocr_result = ocr_result.replace(",", "")
        space_idx = ocr_result.index("\n")
        first = ocr_result[:(space_idx + 1)]
        second = ocr_result[space_idx:]
        second = second.replace("\n", "")
        x_index = second.find("\\x")
        second = second[:x_index]
        temp1 = first + second
        temp2 = ""

        for idx, j in enumerate(temp1):
            if idx > 54 and idx < 58:
                # print("ilk j : "+j)
                j = j.replace("0", "O")
                j = j.replace("1", "I")
                j = j.replace("5", "S")
                j = j.replace("6", "G")
                j = j.replace("8", "B")
                temp2 += j
                # print("son j : "+j)
            else:
                temp2 += j

        ocr_result = temp2

        try:
            check = TD3CodeChecker(ocr_result)
            # print(check)
        except:
            print("LengthError")
            check = False

        if check:
            break


    if not check:
        print("Your MRZ code is not valid!")
    else:
        fields = check.fields()

    mrz_dat = {
        'surname': fields.surname,
        'name': fields.name,
        'sex': fields.sex,
        'birth_date': fields.birth_date,
        'expiry_date': fields.expiry_date,
        'country': get_country(fields.country),
        'nationality': get_country(fields.nationality),
        'document_number': fields.document_number,
        'document_type': fields.document_type
    }

    return mrz_dat


@app.route('/predict', methods=['POST'])
def predict():

    print("Request.method:", request.method)
    print("Request.TYPE", type(request))
    print("In the process of making a prediction.")

    response_dict = {
        'SUCCESS': None,
        'MRZ_Count': None,
        'HEADSHOT_Count': None,
        'MRZ_Data': None,
        'HEADSHOT_Img': None
    }

    global crop_headhsot
    global crop_mrz

    if request.method == 'POST':

        try:
            nparr = np.fromstring(request.data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = np.array(image, dtype=float)

            pass

        except Exception as error:
            print('Error:{}'.format(error))

            response_dict['SUCCESS'] = False
            response_dict['MRZ_Count'] = None
            response_dict['HEADSHOT_Count'] = None
            response_dict['MRZ_Data'] = None
            response_dict['HEADSHOT_Img'] = None

            return response_dict


        try:
            results = model(image, size=640)

            results_df = results.pandas().xyxy[0]
            results_df = results_df[results_df["confidence"] > 0.95]
            results_df = results_df.set_index('name')

            results_df = results_df.values.tolist()
            results_df_length = len(results_df)

            if results_df_length != 0:

                if len(results_df) == 2:

                    class_1 = int(results_df[0][5])
                    class_2 = int(results_df[1][5])

                    if class_1 != class_2:

                        if class_1 == 1:
                            # Headshot
                            crop_headhsot = Crop_Image(image, results_df[0])
                            pass

                        elif class_1 == 0:
                            # MRZ
                            crop_mrz = Crop_Image(image, results_df[0])
                            pass


                        if class_2 == 1:
                            # Headshot
                            crop_headhsot = Crop_Image(image, results_df[1])
                            pass

                        elif class_2 == 0:
                             # MRZ
                             crop_mrz = Crop_Image(image, results_df[1])
                             pass


                        crop_mrz_ocr = Extract_Character(crop_mrz)

                        _, im_arr = cv2.imencode('.jpg', crop_headhsot)
                        im_bytes = im_arr.tobytes()
                        im_b64 = base64.b64encode(im_bytes).decode()

                        response_dict['SUCCESS'] = True
                        response_dict['MRZ_Count'] = 1
                        response_dict['HEADSHOT_Count'] = 1
                        response_dict['MRZ_Data'] = crop_mrz_ocr
                        response_dict['HEADSHOT_Img'] = im_b64

                        pass

                    else:

                        if class_1 == 0:

                            response_dict['MRZ_Count'] = 2
                            response_dict['HEADSHOT_Count'] = 0


                            pass
                        else:

                            response_dict['MRZ_Count'] = 0
                            response_dict['HEADSHOT_Count'] = 2

                            pass
                        pass
                    pass

                elif len(results_df) == 1:

                    mrz_or_headshot = int(results_df[0][5])

                    if mrz_or_headshot == 0:
                        response_dict['MRZ_Count'] = 1
                        response_dict['HEADSHOT_Count'] = 0
                        pass
                    else:
                        response_dict['MRZ_Count'] = 0
                        response_dict['HEADSHOT_Count'] = 1
                        pass

                    pass

                else:

                    for data_list in results_df:

                        if int(data_list[5]) == 0:
                            response_dict['MRZ_Count'] += 1
                            pass
                        else:
                            response_dict['HEADSHOT_Count'] += 1
                            pass

                    pass

            else:
                response_dict['SUCCESS'] = False
                response_dict['MRZ_Count'] = 0
                response_dict['HEADSHOT_Count'] = 0

                return response_dict

                pass

        except Exception as error:
            print('Error:{}'.format(error))

            response_dict['SUCCESS'] = False
            response_dict['MRZ_Count'] = None
            response_dict['HEADSHOT_Count'] = None
            response_dict['MRZ_Data'] = None
            response_dict['HEADSHOT_Img'] = None

            return response_dict


        return flask.jsonify(response_dict)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)


