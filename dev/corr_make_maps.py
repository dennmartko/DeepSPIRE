from pysides.make_cube import gen_radec
from pysides.gen_fluxes_filter import gen_fluxes_filter
from astropy.io import fits
import numpy as np
from astropy import wcs
import astropy.units as u
import datetime
from astropy.stats import gaussian_fwhm_to_sigma
import astropy.convolution as conv
from scipy.signal import convolve
import os

from IPython import embed
import numpy as np


def set_wcs_map(cat, pixel_size):

    #-----------------SET WCS-----------------
    # Coordinate increment at reference point

    if ((not "ra" in cat.columns) or (not "dec" in cat.columns)):
        print("generating the coordinates of the sources")
        ra,dec = gen_radec(cat, params )
    else:
        ra =  np.asarray(cat["ra"]) * u.deg
        dec = np.asarray(cat["dec"]) * u.deg
    
    ra_mean, dec_mean = np.mean(ra.value) , np.mean(dec.value) 

    # set a first time the wcs
    pix_resol = pixel_size / 3600.

    ra_cen = 0.5 * (ra.max() + ra.min())
    dec_cen = 0.5 * (dec.max() + dec.min())
    delta_ra = ra.max() - ra.min()
    delta_dec = dec.max() - dec.min()

    w = wcs.WCS(naxis=2)
    w.wcs.crval = [ra_cen.value, dec_cen.value]
    w.wcs.crpix = [0.5*delta_ra.value / pix_resol, 0.5*delta_dec.value / pix_resol]
    w.wcs.cdelt = [pix_resol, pix_resol]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = [u.deg, u.deg]

    # compute the position in pixel units
    x, y = w.celestial.wcs_world2pix(ra , dec, 0)

    #Offset the central pixel to have all x>=0 and y>=0
    w.wcs.crpix = [0.5*delta_ra.value / pix_resol - np.min(x), 0.5*delta_dec.value / pix_resol  - np.min(y)]

    #recompute x and y in the new WCS
    x, y = w.celestial.wcs_world2pix(ra , dec, 0)

    # list of the position in pixel units: pos[0] = y, pos[1] = x 
    pos = [y , x]
    
    # compute the dimensions of the three axes
    shape = [np.ceil(pos[0].max()), np.ceil(pos[1].max())]
    shape = [i//2*2+1 for i in shape] #force an odd number of pixels to generate better psf
    x_edges = list(np.arange(-0.5, shape[1] + 0.5, 1))
    y_edges = list(np.arange(-0.5, shape[0] + 0.5, 1))

    wcs_dict = {}
    wcs_dict['w'] = w
    wcs_dict['shape'] = shape
    wcs_dict['pos'] = pos
    wcs_dict['x_edges'] = x_edges
    wcs_dict['y_edges'] = y_edges

    return wcs_dict




def save_map(filename, map_array, map_prop_dict, filter_name, unit, beam_fwhm, input_cat):

    print('Write '+filename+'...')
            
    f = fits.PrimaryHDU(map_array, header=map_prop_dict['w'].to_header())
    hdu = fits.HDUList([f])
    hdr = hdu[0].header
    hdr.set("map")
    hdr.set("Datas")
    hdr["BITPIX"] = ("64", "array data type")
    hdr["BUNIT"] = unit
    hdr['COMMENT'] = 'Filter name = '+filter_name
    hdr['COMMENT'] = 'beam FWHM = '+str(beam_fwhm)+' arcsec'
    hdr['COMMENT'] = 'Input catalog = '+input_cat
    hdr["DATE"] = (str(datetime.datetime.now()), "date of creation")
    hdu.writeto(filename, overwrite=True)
    hdu.close()



    
def make_maps(cat, params_maps, params_sides):

    
    flux_filter_list = []
    for filter_name, pixel_size, beam_fwhm in zip(params_maps['filter_list'], params_maps['pixel_size'], params_maps['beam_fwhm_list']):

        print('Generate the map for '+filter_name+'...')

        Sname = 'S'+filter_name
        flux_filter_list.append(Sname)

        #Compute flux list in a filter, if it is missing in the catalog
        if not (Sname in cat.keys()):
            print(filter_name+' fluxes are not included in the catalog. They are computed now...')
            params_temp = params_sides
            params_temp['filter_list'] = [filter_name]
            cat = gen_fluxes_filter(cat, params_temp)

            
        print("Set World Coordinates System...")
        map_prop_dict = set_wcs_map(cat, pixel_size)

        #compute map with beam
        histo, y_edges, x_edges = np.histogram2d(map_prop_dict['pos'][0], map_prop_dict['pos'][1], bins=(map_prop_dict['y_edges'], map_prop_dict['x_edges']), weights=cat['S'+filter_name])

        if os.path.exists(params_maps["output_path"]) == False:
            print('Create '+params_maps["output_path"])
            os.makedirs(params_maps["output_path"])

        #save maps with beams in the requested unit(s)
        if params_maps['gen_map_nobeam_Jy_pix']:
            filename = params_maps["output_path"] + params_maps["run_name"] + '_' + filter_name + '_nobeam_Jy_pix.fits'
            save_map(filename, histo, map_prop_dict, filter_name, 'Jy/pix', 0, params_maps['sides_cat_path'])

        if params_maps['gen_map_nobeam_MJy_sr']:
            pixel_sr = (pixel_size * np.pi/180/3600 )**2 #solid angle of the pixel in sr 
            map_temp =  histo / pixel_sr * 1.e-6 #The 1.e-6 is used to go from Jy to My, final result in MJy/sr
            filename = params_maps["output_path"] + params_maps["run_name"] + '_' + filter_name + '_nobeam_MJy_sr.fits'
            save_map(filename, map_temp, map_prop_dict, filter_name, 'MJy/sr', 0, params_maps['sides_cat_path'])

        #if necessary, compute the beam-smooth map and save in the requested unit(s)
        if params_maps['gen_map_smoothed_Jy_beam'] or params_maps['gen_map_smoothed_MJy_sr']:

            print('Convolve the map by the beam...')
            
            #set the convolution kernel
            sigma_pix =  beam_fwhm * gaussian_fwhm_to_sigma / pixel_size
            
            MIPS24_BEAM_PATH = "DeepSPIRE/data/beams/spitzer-mips24.fits"
            # --- CUSTOM BEAM SPITZER 24 MICRON --- #
            if Sname == 'SMIPS24':
                kernel = fits.open(MIPS24_BEAM_PATH, memmap=False)[0].data
                kernel = conv.CustomKernel(kernel)
                kernel.normalize(mode="peak")
                print("MIPS HERE")
            else:
                kernel_size = 2 * int(np.ceil(20 * sigma_pix)) + 1  # cover ±20σ, make it odd
                print(kernel_size)
                kernel = conv.Gaussian2DKernel(x_stddev=sigma_pix, x_size=kernel_size, y_size=kernel_size)
                kernel.normalize(mode="peak")
                print(Sname + "   ELSE")

            histo_conv = conv.convolve_fft( histo, kernel, normalize_kernel=False, boundary="fill", fill_value=0, allow_huge=True)
            #histo_conv = conv.convolve(histo, kernel, normalize_kernel=False, boundary="fill", fill_value=0)

            if params_maps['gen_map_smoothed_Jy_beam']:
                filename = params_maps["output_path"] + params_maps["run_name"] + '_' + filter_name + '_smoothed_Jy_beam.fits'
                save_map(filename, histo_conv, map_prop_dict, filter_name, 'Jy/beam', 0, params_maps['sides_cat_path'])
                 
            if params_maps['gen_map_smoothed_MJy_sr']:
                conv_factor = np.sum(kernel.array) * (pixel_size  * np.pi/180/3600)**2 * 1.e-6
                print(conv_factor, np.sum(kernel.array), map_prop_dict['shape'][1])
                map_temp = histo_conv / conv_factor
                filename = params_maps["output_path"] + params_maps["run_name"] + '_' + filter_name + '_smoothed_MJy_sr.fits'
                save_map(filename, map_temp, map_prop_dict, filter_name, 'MJy/sr', 0, params_maps['sides_cat_path'])
    #embed()
    #crash

            



