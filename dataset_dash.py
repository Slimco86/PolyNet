import cv2
import plotly
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from io import BytesIO
from PIL import Image
import base64
import os
import numpy as np
import json

data_parameters = ["Person box","Face box","Pose","Landmarks","Race","Gender","Emotion","Skin","Age"]
toggles = [daq.ToggleSwitch(id=param.split(" ")[0],
    value=False,
    label=param,
    labelPosition='top'
) for param in data_parameters]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
fig = go.Figure()
# Add image
img_width = 1600
img_height = 1300
scale_factor = 0.5
fig.add_layout_image(
        x=0,
        sizex=img_width,
        y=0,
        sizey=img_height,
        xref="x",
        yref="y",
        opacity=1.0,
        layer="below",
        source="https://raw.githubusercontent.com/michaelbabyn/plot_data/master/bridge.jpg"
)
fig.update_xaxes(showgrid=False, range=(0, img_width))
fig.update_yaxes(showgrid=False, scaleanchor='x', range=(img_height, 0))

# Set dragmode and newshape properties; add modebar buttons
fig.update_layout(
    dragmode='drawrect',
    newshape=dict(line_color='cyan'),
    title_text='Drag to add annotations - use modebar to change drawing tool'
)

app.layout = html.Div(children=[
                        html.Div([
                            dcc.Input(id="data_path",placeholder="Dataset path",
                                    style={"position":"absolute",
                                            "wdith":"40%"}),
                            html.H4(id='img-name',style={"position":"absolute","top":"10%"}),
                            dcc.Graph(id="img-view",figure=fig,
                                        style={"position":"absolute",
                                                                        "top":"15%",
                                                                        "width":"40%",
                                                                        "height":"70%"},
                                        config={'modeBarButtonsToAdd':['drawline',
                                                'drawopenpath',
                                                'drawclosedpath',
                                                'drawcircle',
                                                'drawrect',
                                                'eraseshape'
                                                        ]}),
                            #html.Img(id="Img-view",src="",
                            #        style={"position":"absolute",
                            #                "top":"15%",
                            #                "wdith":512,
                            #                "height":512,
                            #               }),
                            html.Div(id='btn-cont',children=[                
                                html.Button('Privious', id='prev-but', n_clicks=0),
                                html.Button('Next', id='next-btn', n_clicks=0)],
                                    style={"position":"absolute",
                                            "top":"5%"}),]),
                        html.Div(id="json-data",style={"position":'absolute',
                                                        "top":"80%",
                                                        "width":"60%",
                                                        "left":"15%"}),
                        html.Div(id="Params",children = toggles,
                                style={"position":'absolute',
                                                        "top":"5%",
                                                        "left":"80%"})
])

def numpy_2_b64(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)
    buff = BytesIO()
    im.save(buff,format="png")
    img_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_b64


inputs = [Input(i.id,"value") for i in toggles]
inputs.append(Input("next-btn","n_clicks"))
inputs.append(Input("data_path","value"))

@app.callback([Output('json-data','children'), Output('img-name',"children"), 
            Output("img-view",'figure')],
                inputs)
def update(*args):
    path = args[-1]
    clicks = args[-2]
    files = os.listdir(path)
    names = [x.split('.jpg')[0] for x in files if x.endswith('.jpg')]
    
    name = names[clicks]+'.jpg'
    with open (os.path.join(path,names[clicks]+'_meta.json'),'rb') as f:
        data = json.load(f)
    data_string= json.dumps(data)
    img = cv2.imread(os.path.join(path,name))
    if args[0]:
        boxes = data['person_bbox']
        for box in boxes:
            cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
    if args[1]:
        boxes = data['face_bbox']
        for i,box in enumerate(boxes):
            cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)
            if args[4]:
                cv2.putText(img,str(data['race'][i]),(box[0],box[1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            if args[-1]:
                cv2.putText(img,str(data['age'][i]),(box[0],box[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

    if args[2]:
        landmarks = np.array(data['pose'])
        for p in range(landmarks.shape[0]):
            try:
                for i in range(landmarks.shape[1]):
                    cv2.circle(img,(landmarks[p][i][0],landmarks[p][i][1]),3,(150,255,0),cv2.FILLED)
            except IndexError:
                continue
        

    if args[3]:
        landmarks = np.array(data['face_landmarks'])      
        for p in range(landmarks.shape[0]):
            try:
                for i in range(landmarks.shape[1]):
                    cv2.circle(img,(landmarks[p][i][0],landmarks[p][i][1]),3,(150,0,255),cv2.FILLED)
            except IndexError:
                continue
    
    pil_img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #img = numpy_2_b64(img)
    
    fig = create_figure(pil_img)
    return data_string,name,fig

def create_figure(img):
    fig = go.Figure()
    # Add image
    img_width = img.size[0]
    img_height = img.size[1]
    scale_factor = 0.5
    fig.add_layout_image(
            x=0,
            sizex=img_width,
            y=0,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            source=img
    )
    fig.update_xaxes(showgrid=False, range=(0, img_width))
    fig.update_yaxes(showgrid=False, scaleanchor='x', range=(img_height, 0))
    #Set dragmode and newshape properties; add modebar buttons
    fig.update_layout(
        dragmode='drawrect',
        newshape=dict(line_color='cyan'),
        title_text='Drag to add annotations - use modebar to change drawing tool'
    )

    return fig
if __name__ == '__main__':
    app.run_server(debug=True,port =8051)