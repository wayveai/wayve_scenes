
import dash
import flask
from dash import html
from dash.dependencies import Input, Output
from flask import abort
import os
from dash import dcc, html
from pathlib import Path
import argparse 

from wayve_scenes.scene import WayveScene


# Initialize the Dash app with server
app = dash.Dash(__name__)
app.title = "WayveScenes Viewer"
server = app.server


@server.route('/images/<path:image_path>')
def serve_image(image_path):
    # Securely join the base directory with the image path
    image_directory = os.path.join(BASE_IMAGE_FOLDER, image_path)
    
    # Security check to prevent directory traversal attacks
    if not image_directory.startswith(os.path.abspath(BASE_IMAGE_FOLDER)):
        abort(404)

    # Extract directory and filename
    directory = os.path.dirname(image_directory)
    filename = os.path.basename(image_directory)

    return flask.send_from_directory(directory, filename)



@app.callback(
    Output('image-display', 'children'),
    Input('image-slider', 'value'))
def update_images(selected_set):
    
    timestamp = wayve_scene.unique_timestamps[selected_set]
    
    image_ff = "/images/front-forward/" + str(timestamp) + ".jpeg"
    image_lf = "/images/left-forward/" + str(timestamp) + ".jpeg"
    image_rf = "/images/right-forward/" + str(timestamp) + ".jpeg"
    image_lb = "/images/left-backward/" + str(timestamp) + ".jpeg"
    image_rb = "/images/right-backward/" + str(timestamp) + ".jpeg"
    
    layout = html.Div([
        html.Div([
            html.Img(src=image_lf, style={'width': '33.3%'}),
            html.Img(src=image_ff, style={'width': '33.3%'}),
            html.Img(src=image_rf, style={'width': '33.3%'}),
        ]),
        html.Div([
            html.Img(src=image_lb, style={'width': '50%'}),
            html.Img(src=image_rb, style={'width': '50%'})
        ])
    ])
    
    return layout
    


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('image-slider', 'value'))
def update_scatter_plot(timestamp_index=0):
    return wayve_scene.visualise_colmap_scene(timestamp_index, pointcloud_max_points=100_000)


# Run the app
if __name__ == '__main__':
    
    # parse as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to the root of the dataset. E.g. /path/to/wayve_scenes_101")
    parser.add_argument("--scene_name", type=str, default=None, help="Name of the scene to visualise. E.g. scene_001")
    args = parser.parse_args()
    
    # Set up the paths
    SCENE_DIR = os.path.join(args.dataset_root, args.scene_name)
    BASE_IMAGE_FOLDER = os.path.join(SCENE_DIR, "images")
    COLMAP_PATH = os.path.join(SCENE_DIR, "colmap_sparse", "rig")
    
    # Create scene object
    wayve_scene = WayveScene()

    # Load the scene from disk
    wayve_scene.read_colmap_data(Path(COLMAP_PATH), load_points_camera=False)
        
    # Define Layout of the app
    app.layout = html.Div([
        
        # make div that is centered and 50 % of the screen width        
        html.H1(children='WayveScenes Viewer', style={'textAlign':'center'}),

        html.Div([
            html.H2('Scene: ' + args.scene_name, style={'textAlign':'center'}),
            html.Hr(),
            html.H2('Camera Images', style={'textAlign':'left'}),
            html.H3('Time Slider', style={'textAlign':'left'}),

            dcc.Slider(
                id='image-slider',
                min=0,
                max=len(wayve_scene.unique_timestamps)-1,  
                step=1,
                value=0,
                marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Div(id='image-display'),
            
            html.H2('3D Scene View', style={'textAlign':'left'}),
            html.Div([
                dcc.Graph(id='scatter-plot')
            ])
        ], style={'width': '80%', 'display': 'inline-block'}),
        
    ], style={'width': '90%', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'align-items': 'center', 'backgroundColor': '#ffffff', 'color': '#03b5d1', 'font-family': 'Work Sans Light, sans-serif'})

    app.run_server(debug=True, port=8050)