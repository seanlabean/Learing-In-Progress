import collections
import os
import time
import yt
yt.funcs.mylog.setLevel(40)
import argparse
import multiprocessing
import glob
from functools import partial

parser = argparse.ArgumentParser(description= "Loads in FLASH data output files, \
                                     plots a data field in either a projection or \
                                     slice plot. This process is completed in \
                                     parallel, using all available processor. \
                                     The field value max and minimums can be \
                                     specified across all plots. Values to be set \
                                     by the user: data input and figure output \
                                     directories, field to be plotted, field max/min.\
                                     Assumes that all data files are of the naming \
                                     convention 'turbsph_hdf5_plt_cnt_####'"
                                 )
parser.print_help()
parser.add_argument("-i", "--input_dir", default=None, required=True, type=str,
                    help="Input data directory loc. Prepends /data/draco/slewis")
parser.add_argument("-pt", "--plot_type", default=None, required=True, type=str,
                    help="Plot type. (Options: S(s)lice P(p)rojection, proj)")
parser.add_argument("-ax", "--axis", default=None, required=True, type=str,
                    help="Axis along which perspective of plot is oriented.")
parser.add_argument("-field", "--field_data", default=None, required=True, type=str,
                    help="Data field to be plotted. (e.g. dens, pres, velx)")
parser.add_argument("-fmax", "--max_field_value", default=-1, required=False,
                    type=float, help="Max value of field set on all output plots. \
                    Default is -1.")
parser.add_argument("-fmin", "--min_field_value", default=-1, required=False,
                    type=float, help="Min value of field set on all output plots. \
                    Default is -1.")
parser.add_argument("-z", "--zoom_value", default=-1, required=False, type=int,
                    help="Zoom. Range 1-10. Default is -1 (None).")

def plot_projection(input_dir, ax, field_data, max_field_value, min_field_value, zoom_val, file_name):
    print 'Processor ' + str(os.getpid()) + '\n is plotting projection file ' + str(file_name[-4:])
    ds = yt.load(file_name)
    proj_ = yt.ProjectionPlot(ds, ax, field_data)
    if zoom_val != -1:
        print 'Setting Zoom value.'
        proj_.zoom(zoom_val)
    else:
        print 'No zoom value given. Skipping.'
    if max_field_value != -1:
        proj_.set_zlim(field_data, min_field_value, max_field_value)
    else:
        print 'No max/min field value given. Skipping.'
        None
    proj_.set_font_size(24)
    proj_.annotate_timestamp()
    proj_.annotate_title('$10^5$ M$_{\odot}$ Projection ' + field_data)
    proj_.annotate_scale()
    #proj_.save('/data/draco/slewis' + input_dir + '/figures/' + file_name[-4:])
    proj_.save('./figures/' + file_name[-4:])
    print 'Plot ' + str(file_name[-4:]) + ' saved.'
    
def plot_slice(input_dir, ax, field_data, max_field_value, min_field_value, zoom_val, file_name):
    # print 'Processor ' + str(os.getpid()) + '\n is plotting slice file ' + str(file_name[-4:])
    # ds = yt.load(file_name)
    # print 'Loaded file, making plot now.'
    # slice_ = yt.SlicePlot(ds, ax, field_data)
    # if zoom_val != -1:
    #     print 'Setting Zoom value.'
    #     slice_.zoom(zoom_val)
    # else:
    #     print 'No zoom value given. Skipping.'
    #     None
    # if max_field_value != -1:
    #     print 'I am setting max/min field value'
    #     slice_.set_zlim(field_data, min_field_value, max_field_value)
    # else:
    #     print 'No max/min field value given. Skipping.'
    #     None
    # print 'Setting font size'
    # #slice_.set_font_size(24)
    # print 'Annotating Timestamp'
    # slice_.annotate_timestamp()
    # print 'Adding Title'
    # slice_.annotate_title('$10^5$ M$_{\odot}$ Projection ' + field_data)
    # print 'Adding scale'
    # slice_.annotate_scale()
    # print 'About to save.'
    # slice_.save('/data/draco/slewis' + input_dir + '/figures/' + file_name[-4:])
    # print 'Plot ' + str(file_name[-4:]) + ' saved.'
    print 'Inside plot_slice. On file # ', file_name

start = time.time()

args = parser.parse_args()

plt_type = args.plot_type
data_in = args.input_dir
ax = args.axis
field = args.field_data
max_field = args.max_field_value
min_field = args.min_field_value
zoom = args.zoom_value

pool = multiprocessing.Pool()
#data_files = glob.glob('/data/draco/slewis' + data_in + '/turbsph_hdf5_plt_cnt_*')
data_files = glob.glob('./data/turbsph_hdf5_plt_cnt_*')

func_slice = partial(plot_slice, data_in, ax, field, max_field, min_field, zoom) 
func_proj = partial(plot_projection, data_in, ax, field, max_field, min_field, zoom)
print 'plt type: ', plt_type
print 'axis: ', ax

# Serial plotting, for testing.

#for data in data_files:
#    print 'Using file: '
#    print data
#    plot_slice(data_in, ax, field, max_field, min_field, zoom, data)

if plt_type in ("Slice", "slice", "slc"):
    print plt_type, type(plt_type)
    print 'I am in slice pool.map'
    pool.map(func_slice, data_files)
elif plt_type in ("Projection", "Proj", "projection", "proj"):
    print 'I am in projection pool.map'
    pool.map(func_proj, data_files)
else:
    raise Exception("No acceptable plot type given.")

end = time.time()

print "Total time taken {0:.3f} s".format(end-start)
