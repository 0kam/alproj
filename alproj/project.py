import numpy as np
import moderngl as gl
import math
from PIL import Image
import cv2
import pandas as pd

def projection_mat(fov_x_deg, w, h, near=-1, far=1, cx=None, cy=None):
    if cx == None:
        cx = w/2
    if cy == None:
        cy = h/2
    fov_x = fov_x_deg * math.pi / 180
    fov_y = fov_x * h / w
    fx = 1 / math.tan(fov_x/2)
    fy = 1 / math.tan(fov_y/2)
    mat = np.array([
        fx, 0, (w-2*cx)/w, 0,
        0, fy, -(h-2*cy)/h, 0,
        0, 0, -(far+near)/(far-near), -2*far*near/(far-near),
        0, 0, -1, 0
    ])
    return mat

def modelview_mat(pan_deg, tilt_deg, roll_deg, t_x, t_y, t_z):
    pan = (360-pan_deg) * math.pi / 180
    tilt = tilt_deg * math.pi / 180
    roll = roll_deg * math.pi / 180
    rmat_x = np.array([
        [1, 0, 0, 0],
        [0, math.cos(tilt), -math.sin(tilt), 0],
        [0, math.sin(tilt), math.cos(tilt), 0],
        [0, 0, 0, 1]
    ])
    rmat_y = np.array([
        [math.cos(pan), 0, math.sin(pan), 0],
        [0, 1, 0, 0],
        [-math.sin(pan), 0, math.cos(pan), 0],
        [0, 0, 0, 1]
    ])
    rmat_z = np.array([
        [math.cos(roll), -math.sin(roll), 0, 0],
        [math.sin(roll), math.cos(roll), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    rmat = np.dot(np.dot(rmat_z, rmat_x), rmat_y)
    tmat = np.array([
        [1, 0, 0, -t_x],
        [0, 1, 0, -t_z],
        [0, 0, 1, -t_y],
        [0, 0, 0, 1]
    ])
    return np.dot(rmat, tmat).transpose().flatten()


def persp_proj(vert, value, ind, params):
    ctx = gl.create_standalone_context()
    ctx.enable(gl.DEPTH_TEST) # enable depth testing
    #ctx.enable(gl.CULL_FACE)
    vbo = ctx.buffer(vert.astype("f4").tobytes())
    cbo = ctx.buffer(value.astype("f4").tobytes())
    ibo = ctx.buffer(ind.astype("i4").tobytes()) #vertex indecies of each triangles
    prog = ctx.program(
     vertex_shader='''
         #version 330
         in vec3 in_vert;
         in vec3 in_color;
         out vec3 v_color;  
         // decrare some values used inside GPU by "uniform"
         // the real values will be later set by CPU
         uniform mat4 proj; // projection matrix
         uniform mat4 view; // model view matrix
         uniform float dist_coeffs[14]; // distortion coefficients 
         // distortion coefficients of customized OpenCV model a1, a2, k1~k6, p1, p2, s1~s4
        
         vec4 distort(vec4 view_pos){
          // normalize
          float z =  view_pos[2];
          float z_inv = 1 / z;
          float x1 = view_pos[0] * z_inv;
          float y1 = view_pos[1] * z_inv;
          
          // precalculations
          float x1_2 = x1*x1;
          float y1_2 = y1*y1;
          float x1_y1 = x1*y1;
          float r2 = x1_2 + y1_2;
          float r4 = r2*r2;
          float r6 = r4*r2;
          
          // radial distortion factor
          float r_dist_x = (1+dist_coeffs[2]*r2+dist_coeffs[3]*r4+dist_coeffs[4]*r6) 
                           /(1+dist_coeffs[5]*r2+dist_coeffs[6]*r4+dist_coeffs[7]*r6); 
          float r_dist_y = (1+dist_coeffs[0]+dist_coeffs[2]*r2+dist_coeffs[3]*r4+dist_coeffs[4]*r6) //dist_coeffs[0] = a1
                           /(1+dist_coeffs[1]+dist_coeffs[5]*r2+dist_coeffs[6]*r4+dist_coeffs[7]*r6);//dist_coefs[1] = a2
                          
          // full (radial + tangential + skew) distortion
          float x2 = x1*r_dist_x + 2*dist_coeffs[8]*x1_y1 + dist_coeffs[9]*(r2 + 2*x1_2) + dist_coeffs[10]*r2 + dist_coeffs[11]*r4;
          float y2 = y1*r_dist_y + 2*dist_coeffs[9]*x1_y1 + dist_coeffs[8]*(r2 + 2*y1_2) + dist_coeffs[12]*r2 + dist_coeffs[13]*r4;
          
          // denormalize for projection (which is a linear operation)
          return vec4(x2*z, y2*z, z, view_pos[3]);
          }
         
         void main() {
             vec4 local_pos = vec4(in_vert, 1.0);
             vec4 view_pos = vec4(view * local_pos);
             vec4 dist_pos = distort(view_pos);
             v_color = in_color;
             gl_Position = vec4(proj * dist_pos);
         }
     ''',
     fragment_shader='''
         #version 330
         in vec3 v_color;
         layout(location=0)out vec4 f_color;
         void main() {
             f_color = vec4(v_color, 1.0); // 1,0 added is alpha
         }
     '''
    )
    
    # set some "uniform" values in prog
    proj_mat = projection_mat(params["fov"], params["w"], params["h"])
    view_mat = modelview_mat(params["pan"], params["tilt"], params["roll"], params["x"], params["y"], params["z"])
    dist_coeff = [params["a1"], params["a2"], params["k1"], params["k2"], params["k3"], params["k4"], params["k5"], params["k6"], \
        params["p1"], params["p2"], params["s1"], params["s2"], params["s3"], params["s4"]]
    prog['proj'].value = tuple(proj_mat)
    prog['view'].value = tuple(view_mat)
    prog['dist_coeffs'].value = dist_coeff
    #  pass the vertex, color, index info to the shader
    vao_content = [(vbo, "3f", "in_vert"), (cbo, "3f", "in_color")]
    vao = ctx.vertex_array(program = prog, content = vao_content, index_buffer = ibo)
    # create 2D frame
    rbo = ctx.renderbuffer((params["w"], params["h"]), dtype = "f4")
    drbo = ctx.depth_renderbuffer((params["w"], params["h"]))
    fbo = ctx.framebuffer(rbo, drbo)
    
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    # render the rgb image
    vao.render()
    img = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
    # convert RAW image to ndarray, raw is a 1-dimentional array (rgbrgbrgbrgb......) 
    # array starts from right-bottom of the image, you should flip it in l-r and u-b side
    raw = np.frombuffer((fbo.read(dtype="f4")), dtype = "float32")
    raw = raw.reshape(params["h"], params["w"], 3)
    raw = np.flipud(raw)
    vao.release()
    rbo.release()
    drbo.release()
    fbo.release()
    ctx.release()
    vbo.release()
    cbo.release()
    ibo.release()
    prog.release()
    del(vao_content, vert, value, ind)
    return raw

def sim_image(vert, color, ind, params):
    raw = persp_proj(vert, color, ind, params) * 255
    raw = raw.astype(np.uint8)
    img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    return img

# The shape of array should be (height, width, channel)
def reverse_proj(array, vert, ind, params, chnames=["B", "G", "R"]):
    coord = persp_proj(vert, vert, ind, params)[:, :, [0,2,1]] # channel: x, z, y
    uv = np.meshgrid(np.arange(0,array.shape[1]), np.arange(0,array.shape[0]))
    uv = np.stack(uv, axis = 2)
    concat = np.concatenate([uv, coord, array], 2).reshape(-1, 8)
    columns = ["u", "v", "x", "y", "z"]
    columns.extend(chnames)
    df = pd.DataFrame(concat, columns=columns)
    df[["u", "v"]] = df[["u", "v"]].astype("int16")
    df = df[df["x"] > 0]
    return df

