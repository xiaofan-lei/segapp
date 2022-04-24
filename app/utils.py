# ######################setting up
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace
from azureml.core.model import Model

from configparser import ConfigParser
config = ConfigParser()
configFilePath = r'keys_config.cfg'
config.read(configFilePath)

##authentication with service principal
svc_pr = ServicePrincipalAuthentication(
    tenant_id=config.get('azure-srv', 'tenant_id'),
    service_principal_id=config.get('azure-srv', 'application_id'),
    service_principal_password=config.get('azure-srv', 'svr_password'),
    )
##getting the workspace
ws = Workspace(
    subscription_id=config.get('azure-ws', 'subscription_id'),
    resource_group=config.get('azure-ws', 'resource_group'),
    workspace_name=config.get('azure-ws', 'workspace_name'),
    auth=svc_pr
    )

################# model retrieving################"
import numpy as np
import tensorflow as tf
import cv2
import segmentation_models as sm
import base64
from io import BytesIO
from matplotlib.figure import Figure

class PSPNetInferrer:
    def __init__(self, img_size=(384, 384)):
        self.saved_path = Model.get_model_path(model_name='segmentation_model', version=1, _workspace=ws)
        self.model = tf.keras.models.load_model(self.saved_path, compile=False)
        self.img_size = img_size

    def infer(self, img_data):
        self.img_data = img_data
        image = cv2.resize(self.img_data, self.img_size, interpolation=cv2.INTER_AREA)

        prep = sm.get_preprocessing('efficientnetb3')
        image = prep(image)

        img_array = np.zeros((1,) + self.img_size + (3,), dtype="float32")
        img_array[0] = image

        return self.model.predict(img_array)

################# mask plotting################"
def mask_to_labelIds(pred):
    pred=cv2.resize(pred.squeeze(), (2048,1024), interpolation = cv2.INTER_AREA)
    mask = np.argmax(pred, axis=-1)
    return mask

def mask_plot(mask):
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    ax.imshow(mask,'gray')

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return data

def upload_file(mask_file_path, TARGET_PATH):
    # Getting workspace default datastore
    datastore = ws.get_default_datastore()
    datastore.upload_files(
        # List[str] of absolute paths of files to upload
        [mask_file_path],
        #'<path/on/datastore>'
        target_path=TARGET_PATH,
        overwrite=True,
        )

