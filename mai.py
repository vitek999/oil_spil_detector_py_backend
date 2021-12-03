from flask import Flask, jsonify, request
from sentinelhub import SHConfig
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, DataCollection
import numpy as np

CLIENT_ID = ''
CLIENT_SECRET = ''
INSTANCE_ID = ''

config = SHConfig()

if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET
    config.instance_id = INSTANCE_ID

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

evalscript_all_bands = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
                units: "DN"
            }],
            output: {
                bands: 13,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B01, 
                sample.B02, 
                sample.B03, 
                sample.B04, 
                sample.B05, 
                sample.B06, 
                sample.B07, 
                sample.B08, 
                sample.B8A, 
                sample.B09, 
                sample.B10, 
                sample.B11, 
                sample.B12];
    }
"""


def get_image_data(bbox, height, width, start_date, end_date):
    request_size = (width, height)
    request_bbox = BBox(bbox=bbox, crs=CRS.WGS84)

    request_all_bands = SentinelHubRequest(
        evalscript=evalscript_all_bands,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(start_date, end_date),
                mosaicking_order='leastCC'
            )],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=request_bbox,
        size=request_size,
        config=config
    )

    all_bands_response = request_all_bands.get_data()
    return all_bands_response


app = Flask(__name__)


@app.route('/image', methods=['GET'])
def get_image():
    request_height = request.args.get('height', 64)
    request_width = request.args.get('width', 64)
    request_bbox = request.args.get('bbox').split(',')  #[73.03322551561014, 60.588367153736506, 73.04455362140644, 60.59428130212325]
    request_date_start = request.args.get('date_start') # '2020-06-01'
    request_date_end = request.args.get('date_end') # '2020-06-30'
    res = get_image_data(request_bbox, request_height, request_width,
                         request_date_start, request_date_end)
    print(res[0].shape)
    return jsonify({"data": np.asarray(res).tolist()})


if __name__ == '__main__':
    app.run(port=8081)

# print(get_image_data([73.03322551561014, 60.588367153736506, 73.04455362140644, 60.59428130212325], 64, 64,
#                      test_start_date, test_end_date)[0].shape)
