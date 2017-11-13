/*  mincbeast.c
 *
 *  Copyright 2011  Simon Fristed Eskildsen, Vladimir Fonov,
 *                Pierrick Coupé, Jose V. Manjon
 *
 *  This file is part of mincbeast.
 *
 *  mincbeast is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  mincbeast is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with mincbeast.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  For questions and feedback, please contact:
 *  Simon Fristed Eskildsen <eskild@gmail.com>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif //HAVE_CONFIG_H

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#include "ParseArgv.h"
#include "array_alloc.h"
#include "nlmseg.h"
#include "beast.h"
#include "label.h"

const char LICENSE[]="Copyright (C) 2011\tSimon Fristed Eskildsen, Vladimir S. Fonov, \n\
\t\t\tPierrick Coupe, Jose V. Manjon\n\n\
This program comes with ABSOLUTELY NO WARRANTY; for details type 'cat COPYING'. \n\
This is free software, and you are welcome to redistribute it under certain\n\
conditions; type 'cat COPYING' for details.\n\
";

const char REFERENCE[]="Reference: \n\tEskildsen SF, Coupe P, Fonov V, Manjon JV, Leung KK,\n\tGuizard N, Wassef SN, Ostergaard LR, Collins DL;\n\tAlzheimer's Disease Neuroimaging Initiative.\n\
\tBEaST: brain extraction based on nonlocal segmentation technique.\n\
\tNeuroimage. 2012 Feb 1;59(3):2362-73.\n\
\thttp://dx.doi.org/10.1016/j.neuroimage.2011.09.012\n\n";

#ifdef MT_USE_OPENMP
    #include <omp.h>
#else
    #define omp_get_num_threads() 1
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
#endif

struct beast_options
{
  char *input_file;
  char *output_file;
  char *libdir;
  char *history_label;
  
  double lambda1;
  double lambda2;
  int    sparse_mode;
  
  int sparse_stride;

  VIO_BOOL outputprob;
  VIO_BOOL flipimages;
  VIO_BOOL load_moments;
  VIO_BOOL fill_output;
  VIO_BOOL verbose;
  VIO_BOOL medianfilter;
  VIO_BOOL patchfilter;
  VIO_BOOL abspath;
  VIO_BOOL same_res;
  VIO_BOOL clobber    ;
  VIO_BOOL nomask     ;
  VIO_BOOL nopositive ;
  VIO_BOOL use_sparse;

  int voxelsize;
  int sizepatch;
  int searcharea;
  double alpha;
  double beta;
  double threshold;
  int selectionsize;
  
  char *positive_file;
  char *selection_file;
  char *count_file;
  char *conf_file;
  char *mask_file;
  char *library_prefix;
};


int get_arguments(int argc, char  *argv[] , beast_options * _options)
{
  _options->sparse_stride=1;
  _options->lambda1      = 0.15;
  _options->lambda2      = 0.0;
  _options->sparse_mode  = 2;
  _options->outputprob   = FALSE;
  _options->flipimages   = FALSE;
  _options->load_moments = FALSE;
  _options->fill_output  = FALSE;
  _options->verbose      = FALSE;
  _options->medianfilter = FALSE;
  _options->patchfilter  = FALSE;
  _options->abspath      = FALSE;
  _options->same_res     = TRUE;
  _options->clobber      = FALSE;
  _options->nomask       = FALSE;
  _options->nopositive   = FALSE;
  _options->use_sparse   = FALSE;
  _options->voxelsize    = 4;
  _options->sizepatch    = 1;
  _options->searcharea   = 2;
  _options->alpha        = 0.5;
  _options->beta         = 0.25;
  _options->threshold    = 0.95;
  _options->selectionsize = 20;
  _options->positive_file = NULL;
  _options->selection_file= NULL;
  _options->count_file  = NULL;
  _options->conf_file   = NULL;
  _options->mask_file   = NULL;
  _options->library_prefix = "library";
  
  
  /* Argument table */
  ArgvInfo argTable[] = {
    {
      "-probability", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->outputprob,
      "Output the probability map instead of crisp mask."
    },
    {
      "-flip", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->flipimages,
      "Flip images around the mid-sagittal plane to increase patch count."
    },
    {
      "-load_moments", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->load_moments,
      "Do not calculate moments instead use precalculated library moments. (for optimization purposes)"
    },
    {
      "-fill", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->fill_output,
      "Fill holes in the binary output."
    },
    {
      "-median", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->medianfilter,
      "Apply a median filter on the probability map."
    },
    {
      "-nlm_filter", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->patchfilter,
      "Apply an NLM filter on the probability map (experimental)."
    },
    {
      "-verbose", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->verbose,
      "Enable verbose output."
    },
    {
      "-clobber", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->clobber,
      "Clobber output files"
    },
    {
      "-abspath", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->abspath,
      "File paths in the library are absolute (default is relative to library root)."
    },

    {
      "-voxel_size", ARGV_INT, (char *) 1, (char *) &_options->voxelsize,
      "Specify voxel size for calculations (4, 2, or 1). Assumes no multiscale. Use configuration file for multiscale."
    },
    {
      "-patch_size", ARGV_INT, (char *) 1, (char *) &_options->sizepatch,
      "Specify patch size for single scale approach."
    },
    {
      "-search_area", ARGV_INT, (char *) 1, (char *) &_options->searcharea,
      "Specify size of search area for single scale approach."
    },
    {
      "-alpha", ARGV_FLOAT, (char *) 1, (char *) &_options->alpha,
      "Specify confidence level Alpha."
    },
    {
      "-beta", ARGV_FLOAT, (char *) 1, (char *) &_options->beta,
      "Specify smoothness factor Beta."
    },
    {
      "-threshold", ARGV_FLOAT, (char *) 1, (char *) &_options->threshold,
      "Specify threshold for patch selection."
    },
    {
      "-selection_num", ARGV_INT, (char *) 1, (char *) &_options->selectionsize,
      "Specify number of selected images."
    },

    {
      "-positive", ARGV_STRING, (char *) 1, (char *) &_options->positive_file,
      "Specify mask of positive segmentation (inside mask) instead of the default mask."
    },
    {
      "-output_selection", ARGV_STRING, (char *) 1, (char *) &_options->selection_file,
      "Specify file to output selected files."
    },
    {
      "-count", ARGV_STRING, (char *) 1, (char *) &_options->count_file,
      "Specify file to output the patch count."
    },
    {
      "-configuration", ARGV_STRING, (char *) 1, (char *) &_options->conf_file,
      "Specify configuration file."
    },
    {
      "-mask", ARGV_STRING, (char *) 1, (char *) &_options->mask_file,
      "Specify a segmentation mask instead of the the default mask."
    },
    {
      "-same_resolution", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->same_res,
      "Output final mask with the same resolution as input file."
    },
    {
      "-no_same_resolution", ARGV_CONSTANT, (char *) FALSE, (char *) &_options->same_res,
      "Output final mask downsampled at processing resolution."
    },
    {
      "-no_mask", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->nomask,
      "Do not apply a segmentation mask. Perform the segmentation over the entire image."
    },
    {
      "-no_positive", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->nopositive,
      "Do not apply a positive mask."
    },
#ifdef USE_SPAMS
    {
      "-sparse", ARGV_CONSTANT, (char *) TRUE, (char *) &_options->use_sparse,
      "Use sparse patch merging."
    },
    {
      "-lambda1", ARGV_FLOAT, (char *) 1, (char *) &_options->lambda1,
      "Sparsity cost lambda1."
    },
    {
      "-lambda2", ARGV_FLOAT, (char *) 1, (char *) &_options->lambda2,
      "Sparsity cost lambda2."
    },
    {
      "-sparse_mode", ARGV_INT, (char *) 1, (char *) &_options->sparse_mode,
      "Sparse mode."
    },
    
    {
      "-stride", ARGV_INT, (char *) 1, (char *) &_options->sparse_stride,
      "Stride for spars segmentation speedup with possible quality degradation. (DON'T USE!)"
    },
#endif
    //library_prefix
    {
      "-library_prefix", ARGV_STRING, (char *) 1, (char *) &_options->library_prefix,
      "library prefix, for cross-validation experiment."
    },
    {NULL, ARGV_END, NULL, NULL, NULL}
  };
  
  _options->history_label=create_minc_timestamp(argc,argv);

  /* Get arguments */
  if ( ParseArgv(&argc, argv, argTable, 0) || (argc < 4) ) {
    fprintf(stderr,LICENSE);
    fprintf(stderr,REFERENCE);
    fprintf(stderr,
            "\nUsage: %s [options] <library dir> <input> <output>\n",
            argv[0]);
    fprintf(stderr,"       %s -help\n\n", argv[0]);

    return STATUS_ERR;
  }

  _options->libdir      = argv[argc-3];
  _options->input_file  = argv[argc-2];
  _options->output_file = argv[argc-1];
  
  return STATUS_OK;
}


void cleanup_arguments(beast_options * _options)
{
  free(_options->history_label);
  
  if(_options->mask_file)
    free(_options->mask_file);
  if(_options->positive_file)
    free(_options->positive_file);
}

int main(int argc, char  *argv[] )
{
  beast_options _options;
  
  char imagelist[FILENAMELENGTH], masklist[FILENAMELENGTH], meanlist[FILENAMELENGTH], varlist[FILENAMELENGTH];
  char ***images, ***masks,***means,***vars;
  int num_images,i,sizes[3][5],tmpsizes[5],volumesize,*selection,steps=3,filled=0;
  float *imagedata,*maskdata,*meandata,*vardata,**subject,**mask,**positivemask=NULL,**segsubject,**patchcount,**filtered;
  float max,min;
  float **segmented;
  float *tempdata;
  
  int scale, scaledvolumesize, scales[3] = {1,2,4};
  int masksize=0, initialscale, targetscale, scalesteps;
  
  beast_conf input_conf[3],configuration[3];
  image_metadata **meta;
  image_metadata *mask_meta;
  image_metadata *temp_meta;
  time_t timer;


  const char *default_beast_library=BEAST_LIBRARY_PREFIX;
  const char *default_beast_mask=BEAST_LIBRARY_PREFIX"/margin_mask.mnc";
  const char *default_beast_positive_file=BEAST_LIBRARY_PREFIX"/intersection_mask.mnc";
  const char *default_beast_config=BEAST_LIBRARY_PREFIX"/default.2mm.conf";

  
  fprintf(stderr,"\nmincbeast --\t\tan implementation of BEaST (Brain Extraction\n\t\t\tusing non-local Segmentation Technique) version %s\n\n",PACKAGE_VERSION);
  
#ifdef MT_USE_OPENMP
  fprintf(stderr,"Using OpenMP, max number of threads=%d\n",omp_get_max_threads());
#endif
  
  /* Get the time, overwriting newline */
  timer = time(NULL);
  
  if( get_arguments(argc, argv, &_options)!=STATUS_OK )
    return STATUS_ERR;
  

  if (_options.mask_file==NULL) {
    _options.mask_file=(char*)malloc((strlen(_options.libdir)+20)*sizeof(*_options.mask_file));
    sprintf(_options.mask_file,"%s/margin_mask.mnc",_options.libdir);
  }
  if ((!_options.nopositive) && (_options.positive_file==NULL)) {
    _options.positive_file=(char*)malloc((strlen(_options.libdir)+30)*sizeof(*_options.positive_file));
    sprintf(_options.positive_file,"%s/intersection_mask.mnc",_options.libdir);
  }

  if(!_options.clobber) {
    if(!access(_options.output_file,F_OK)) {
      fprintf(stderr,"ERROR! File exists: %s , run with -clobber\n",_options.output_file);
      return STATUS_ERR;
    }
    if(_options.count_file && !access(_options.count_file,F_OK)) {
      fprintf(stderr,"ERROR! File exists: %s , run with -clobber\n",_options.count_file);
      return STATUS_ERR;
    }
  }

  if ((_options.voxelsize>4) || (_options.voxelsize<1) || (_options.voxelsize==3)) {
    fprintf(stderr,"ERROR! Initial voxel size must be either 4, 2, or 1\n");
    return STATUS_ERR;
  }

  meta = (image_metadata **)malloc(3*sizeof(image_metadata*));

  meta[0] = read_volume(_options.input_file, &tempdata, sizes[0]);
  if (meta[0] == NULL) {
    fprintf(stderr,"ERROR! Image not read: %s\n",_options.input_file);
    return STATUS_ERR;
  }
  volumesize=sizes[0][0]*sizes[0][1]*sizes[0][2];

  subject = alloc_2d_float(3,volumesize); /*VF:memory waste....*/
  cp_volume(tempdata, subject[0], sizes[0]);
  free(tempdata);

  if ((temp_meta=read_volume(_options.mask_file, &tempdata, tmpsizes)) == NULL) {
    fprintf(stderr,"ERROR! Image not read: %s\n",_options.mask_file);
    return STATUS_ERR;
  }
  free_meta(temp_meta);

  if ((tmpsizes[0]!=sizes[0][0]) || (tmpsizes[1]!=sizes[0][1]) || (tmpsizes[2]!=sizes[0][2])) {
    fprintf(stderr,"ERROR! Mask dimension does not match image dimension!\n");
    return STATUS_ERR;
  }
  mask = alloc_2d_float(3,volumesize);
  cp_volume(tempdata, mask[0], sizes[0]);
  free(tempdata);

  if (_options.nomask) {
    /* option for no segmentation mask - set the mask to all ones */
    wipe_data(mask[0],sizes[0],1.0);
  }

  if (_options.positive_file!=NULL) {
    image_metadata *positive_meta;
    if ((positive_meta=read_volume(_options.positive_file, &tempdata, tmpsizes)) == NULL) {
      fprintf(stderr,"ERROR! Image not read: %s\n",_options.positive_file);
      return STATUS_ERR;
    }
    if ((tmpsizes[0]!=sizes[0][0]) || (tmpsizes[1]!=sizes[0][1]) || (tmpsizes[2]!=sizes[0][2])) {
      fprintf(stderr,"ERROR! Positive mask dimension does not match image dimension!\n");
      return STATUS_ERR;
    }
    positivemask = alloc_2d_float(3,volumesize);
    cp_volume(tempdata, positivemask[0], sizes[0]);
    free(tempdata);
    free_meta(positive_meta);

    down_sample(positivemask[0], positivemask[1], 2, sizes[0]);
    down_sample(positivemask[0], positivemask[2], 4, sizes[0]);
  }

  segmented = alloc_2d_float(3,volumesize);

  /* downsample the subject and mask */
  down_sample(subject[0], subject[1], 2, sizes[0]);
  down_sample(subject[0], subject[2], 4, sizes[0]);
  down_sample(mask[0], mask[1], 2, sizes[0]);
  down_sample(mask[0], mask[2], 4, sizes[0]);

  /* populate the entire configuration table for compatibility reasons */
  for (i=0; i<3; i++) {
    configuration[i].voxelsize = _options.voxelsize;
    configuration[i].patchsize = _options.sizepatch;
    configuration[i].searcharea = _options.searcharea;
    configuration[i].alpha = _options.alpha;
    configuration[i].beta = _options.beta;
    configuration[i].threshold = _options.threshold;
    configuration[i].selectionsize = _options.selectionsize;
  }


  if (_options.conf_file != NULL) {
    steps=read_configuration(_options.conf_file, input_conf);
    if (steps==STATUS_ERR) {
      fprintf(stderr,"Error in configuration file. Values outside limits.\n");
      return STATUS_ERR;
    }
    initialscale=-1;
    targetscale=4;
    for (i=0; i<steps; i++) {
      scale=(int)(input_conf[i].voxelsize/2);
      configuration[scale].voxelsize=input_conf[i].voxelsize;
      configuration[scale].patchsize=input_conf[i].patchsize;
      configuration[scale].searcharea=input_conf[i].searcharea;
      configuration[scale].alpha=input_conf[i].alpha;
      configuration[scale].beta=input_conf[i].beta;
      configuration[scale].threshold=input_conf[i].threshold;
      configuration[scale].selectionsize=input_conf[i].selectionsize;
      if (scale>initialscale)
        initialscale=scale;
      if (scale<targetscale)
        targetscale=scale;
    }
  } else {
    /* if no configuration file, apply user settings for single scale */
    targetscale=initialscale=(int)(_options.voxelsize/2);
  }

  scalesteps=initialscale-targetscale+1;

  fprintf(stderr,"%d scale steps:\n",scalesteps);

  for (i=initialscale; i>=targetscale; i--) {
    fprintf(stderr,"%d %d %d %4.2lf %4.2lf %4.2lf %d\n",
            configuration[i].voxelsize, configuration[i].patchsize, configuration[i].searcharea,
            configuration[i].alpha,     configuration[i].beta, configuration[i].threshold, configuration[i].selectionsize);
  }

  images = alloc_3d_char(3, MAXLIBSIZE, FILENAMELENGTH);
  masks =  alloc_3d_char(3, MAXLIBSIZE, FILENAMELENGTH);
  means =  alloc_3d_char(3, MAXLIBSIZE, FILENAMELENGTH);
  vars =   alloc_3d_char(3, MAXLIBSIZE, FILENAMELENGTH);

  /*for (scale=initialscale;scale>=0;scale--){*/
  for (scale=2; scale>=0; scale--) {

    sprintf(imagelist,"%s/%s.stx.%dmm", _options.libdir, _options.library_prefix, scales[scale]);
    sprintf(masklist,"%s/%s.masks.%dmm",_options.libdir, _options.library_prefix, scales[scale]);
    if (_options.load_moments) {
      sprintf(meanlist,"%s/%s.means.%dmm",_options.libdir, _options.library_prefix, scales[scale]);
      sprintf(varlist,"%s/%s.vars.%dmm",  _options.libdir, _options.library_prefix, scales[scale]);
    }
    num_images=read_list( imagelist, images[scale], _options.abspath?"":_options.libdir );
    if (read_list( masklist, masks[scale], _options.abspath?"":_options.libdir )!=num_images) {
      fprintf(stderr,"ERROR! Number of images and masks does not match!\n");
      return STATUS_ERR;
    }

    if ( num_images<configuration[scale].selectionsize ) {
      fprintf(stderr,"ERROR! Cannot select more images than in the library!\n\tlibrary images: %d\n\tselection: %d\n",num_images,configuration[scale].selectionsize);
      return STATUS_ERR;
    }

    if ( _options.load_moments ) {
      if ( read_list(meanlist, means[scale],_options.abspath?"":_options.libdir)!=num_images ) {
        fprintf(stderr,"ERROR! Number of images and means does not match!\n");
        return STATUS_ERR;
      }
      if ( read_list(varlist, vars[scale],_options.abspath?"":_options.libdir)!=num_images ) {
        fprintf(stderr,"ERROR! Number of images and vars does not match!\n");
        return STATUS_ERR;
      }
    }
  }

  if ((mask_meta=read_volume(_options.mask_file, &tempdata, tmpsizes)) == NULL) {
    fprintf(stderr,"ERROR! Image not read: %s\n",_options.mask_file);
    return STATUS_ERR;
  }
  if ((tmpsizes[0]!=sizes[0][0]) || (tmpsizes[1]!=sizes[0][1]) || (tmpsizes[2]!=sizes[0][2])) {
    fprintf(stderr,"ERROR! Image dimension does not match library image dimension!\n");
    return STATUS_ERR;
  }
  free(tempdata);
  free_meta(mask_meta);

  meta[1] = read_volume(images[1][0], &tempdata, sizes[1]);
  if (meta[1] == NULL) {
    fprintf(stderr,"ERROR! Image not read: %s\n",images[1][0]);
    return STATUS_ERR;
  }
  free(tempdata);

  meta[2] = read_volume(images[2][0], &tempdata, sizes[2]);
  if (meta[2] == NULL) {
    fprintf(stderr,"ERROR! Image not read: %s\n",images[2][0]);
    return STATUS_ERR;
  }
  free(tempdata);

  /* make the downsampled masks crisp */
  threshold_data( mask[1], sizes[1], 0.5);
  threshold_data( mask[2], sizes[2], 0.5);

  segsubject = alloc_2d_float(3, volumesize);
  patchcount = alloc_2d_float(3, volumesize);
  filtered   = alloc_2d_float(3, volumesize);

  if (_options.verbose) fprintf(stderr,"Initial voxel size: %d\nTarget voxel size: %d\n", scales[initialscale], scales[targetscale]);

  for (scale=initialscale; scale>=targetscale; scale--) {
    int selection_size=configuration[scale].selectionsize;
    
    selection = (int *)malloc(configuration[scale].selectionsize*sizeof(*selection));
    pre_selection(subject[scale], mask[scale], images[scale], sizes[scale], num_images, 
                  configuration[scale].selectionsize, selection, _options.selection_file,_options.verbose);

    if (_options.verbose) fprintf(stderr,"Performing segmentation at %dmm resolution\nReading files ",scales[scale]);

    scaledvolumesize = sizes[scale][0]*sizes[scale][1]*sizes[scale][2];

    imagedata = (float *)malloc(configuration[scale].selectionsize*scaledvolumesize*sizeof(float));
    maskdata =  (float *)malloc(configuration[scale].selectionsize*scaledvolumesize*sizeof(float));
    meandata =  (float *)malloc(configuration[scale].selectionsize*scaledvolumesize*sizeof(float));
    vardata =   (float *)malloc(configuration[scale].selectionsize*scaledvolumesize*sizeof(float));

    /* read the library images, masks, and moments */
    for (i=0; i<configuration[scale].selectionsize; i++) {
      image_metadata *_meta;
      if (_options.verbose) fprintf(stderr,".");
      if ((_meta=read_volume(images[scale][selection[i]], &tempdata, tmpsizes)) == NULL) {
        fprintf(stderr,"ERROR! Image not read: %s\n",images[scale][selection[i]]);
        return STATUS_ERR;
      }
      cp_volume(tempdata, imagedata+i*scaledvolumesize, tmpsizes);
      free(tempdata);
      free_meta(_meta);
    }
    if (_options.verbose) fprintf(stderr,"*");
    for (i=0; i<configuration[scale].selectionsize; i++) {
      image_metadata *_meta;
      if (_options.verbose) fprintf(stderr,".");
      if ((_meta=read_volume(masks[scale][selection[i]], &tempdata, tmpsizes)) == NULL) {
        fprintf(stderr,"ERROR! Image not read: %s\n",masks[scale][selection[i]]);
        return STATUS_ERR;
      }
      cp_volume(tempdata, maskdata+i*scaledvolumesize, tmpsizes);
      free(tempdata);
      free_meta(_meta);
    }
    if (_options.verbose) fprintf(stderr,"*");

    if (!_options.load_moments) {
      /* calculate the mean and variance for the library images */
      /* this must be done if the selected patch size is different from the one used in the precalculation */
      for (i=0; i<configuration[scale].selectionsize; i++) {
        if (_options.verbose) fprintf(stderr,"c");
        ComputeFirstMoment(imagedata+i*scaledvolumesize, meandata+i*scaledvolumesize, sizes[scale], configuration[scale].patchsize, &min, &max);
        ComputeSecondMoment(imagedata+i*scaledvolumesize, meandata+i*scaledvolumesize, vardata+i*scaledvolumesize, sizes[scale], configuration[scale].patchsize, &min, &max);
      }
    } else {
      for (i=0; i<configuration[scale].selectionsize; i++) {
        image_metadata *_meta;
        if (_options.verbose) fprintf(stderr,".");
        if ((_meta=read_volume(means[scale][selection[i]], &tempdata, tmpsizes)) == NULL) {
          fprintf(stderr,"ERROR! Image not read: %s\n",means[scale][selection[i]]);
          return STATUS_ERR;
        }
        cp_volume(tempdata, meandata+i*scaledvolumesize, tmpsizes);
        free(tempdata);
        free_meta(_meta);
      }
      if (_options.verbose) fprintf(stderr,"*");
      for (i=0; i<configuration[scale].selectionsize; i++) {
        image_metadata *_meta;
        fprintf(stderr,".");
        if ((_meta=read_volume(vars[scale][selection[i]], &tempdata, tmpsizes)) == NULL) {
          fprintf(stderr,"ERROR! Image not read: %s\n",masks[scale][selection[i]]);
          return STATUS_ERR;
        }
        cp_volume(tempdata, vardata+i*scaledvolumesize, tmpsizes);
        free(tempdata);
        free_meta(_meta);
      }
    }
    if (_options.verbose) fprintf(stderr,"\n");
    /* end of reading files */

    /* remove any disconnected parts */
    masksize = getLargestObject_float(mask[scale], sizes[scale], 1, 0);

    if (_options.verbose) fprintf(stderr,"Mask size: %d\nAlpha: %f\n",masksize,configuration[scale].alpha);

    /* make sure we starting from a clean slate */
    wipe_data(segsubject[scale],sizes[scale],0.0);

    if (_options.flipimages) {
      /* doubling the library selection by flipping images along the mid-sagittal plane */
      imagedata = (float *)realloc(imagedata,configuration[scale].selectionsize*2*scaledvolumesize*sizeof(*imagedata));
      maskdata =  (float *)realloc(maskdata, configuration[scale].selectionsize*2*scaledvolumesize*sizeof(*maskdata) );
      meandata =  (float *)realloc(meandata, configuration[scale].selectionsize*2*scaledvolumesize*sizeof(*meandata) );
      vardata =   (float *)realloc(vardata,  configuration[scale].selectionsize*2*scaledvolumesize*sizeof(*vardata)  );

      for (i=0; i<configuration[scale].selectionsize; i++) {
        flip_data(imagedata+i*scaledvolumesize, imagedata+(configuration[scale].selectionsize+i)*scaledvolumesize, sizes[scale]);
        flip_data(maskdata+i*scaledvolumesize,  maskdata+(configuration[scale].selectionsize+i)*scaledvolumesize,  sizes[scale]);
        flip_data(meandata+i*scaledvolumesize,  meandata+(configuration[scale].selectionsize+i)*scaledvolumesize,  sizes[scale]);
        flip_data(vardata+i*scaledvolumesize,   vardata+(configuration[scale].selectionsize+i)*scaledvolumesize,   sizes[scale]);
      }
      selection_size=configuration[scale].selectionsize*2;
    }
    if(_options.use_sparse) {
#ifdef USE_SPAMS    
        max = nlmsegSparse4D(subject[scale], imagedata, maskdata, meandata, vardata, mask[scale], 
                        configuration[scale].patchsize, configuration[scale].searcharea, configuration[scale].beta, 
                        configuration[scale].threshold, sizes[scale], selection_size, segsubject[scale], patchcount[scale],
                        _options.lambda1,_options.lambda2,_options.sparse_mode,_options.sparse_stride
                            );
#endif
    }  else {
        max = nlmsegFuzzy4D(subject[scale], imagedata, maskdata, meandata, vardata, mask[scale], 
                        configuration[scale].patchsize, configuration[scale].searcharea, configuration[scale].beta, 
                        configuration[scale].threshold, sizes[scale], selection_size, segsubject[scale], patchcount[scale]);
    }

    free(imagedata);
    free(maskdata);
    free(meandata);
    free(vardata);


    if ( _options.positive_file!=NULL ) {
      /* add the certain positive segmentation (inside mask) */
      add_mask_data(segsubject[scale], positivemask[scale], sizes[scale]);
    }

    /* add the certain segmentation from the previous scale */
    add_mask_data(segsubject[scale], segmented[scale], sizes[scale]);

    if ( _options.medianfilter ) {
      median_filter(segsubject[scale], sizes[scale], 3);
    }

    /* the patch filter is experimental */
    if ( _options.patchfilter ) {
      wipe_data(filtered[scale],sizes[scale],0.0);
      wipe_data(patchcount[scale],sizes[scale],0.0);
      max = nlmfilter(subject[scale], mask[scale], segsubject[scale], 2*configuration[scale].patchsize, 2*configuration[scale].searcharea, configuration[scale].beta, configuration[scale].threshold, sizes[scale], filtered[scale], patchcount[scale]);
      combine_maps(segsubject[scale], filtered[scale], mask[scale], sizes[scale]);
    }

    if (scale > targetscale) {
      /* if performing a higher resolution step, upsample the result and create new mask */
      resize_trilinear(segsubject[scale], sizes[scale], sizes[scale-1], segsubject[scale-1]);
      
      masksize=update_mask(segsubject[scale-1], mask[scale-1], segmented[scale-1], sizes[scale-1], 
                           configuration[scale].alpha, 1.0-configuration[scale].alpha);
    }

    free(selection);
  } /* for each scale */


  if (_options.count_file!=NULL) {
    if(write_volume_generic(_options.count_file, patchcount[targetscale], meta[targetscale],FALSE))
    {
      fprintf(stderr,"Can't save output to %s\n",_options.count_file);
      return STATUS_ERR;
    }
  }

  if(targetscale!=0 && _options.same_res) { /* need to upsample final output */
    if (_options.verbose) fprintf(stderr,"Upsampling to input resolution, %dx%dx%d\n",sizes[0][0],sizes[0][1],sizes[0][2]);
    resize_trilinear(segsubject[targetscale], sizes[targetscale], sizes[0], segsubject[0]);
    masksize=update_mask(segsubject[0], mask[0], segmented[0], sizes[0], configuration[targetscale].alpha, 1.0-configuration[targetscale].alpha);
    targetscale=0;
    configuration[targetscale].alpha = _options.alpha;
  }

  if (!_options.outputprob) {
    if (_options.verbose) fprintf(stderr,"Thresholding estimator at %f\n",configuration[targetscale].alpha);
    threshold_data(segsubject[targetscale], sizes[targetscale], configuration[targetscale].alpha);
    getLargestObject_float(segsubject[targetscale], sizes[targetscale], 1, 0);

    if (_options.fill_output) {
      wipe_data(mask[targetscale], sizes[targetscale], 1.0);
      filled = flood_fill_float(segsubject[targetscale], mask[targetscale], sizes[targetscale], 0, 0, 0, 0, 6);
      //segsubject[targetscale]=mask[targetscale];
      cp_volume(mask[targetscale],segsubject[targetscale],sizes[targetscale]);
    }
  }

  meta[targetscale]->history=strdup(_options.history_label);
  if(write_volume_generic(_options.output_file, segsubject[targetscale], meta[targetscale],!_options.outputprob)) {
    fprintf(stderr,"Can't save output to %s\n", _options.output_file);
    return STATUS_ERR;
  }

  free_2d_float(mask);
  free_2d_float(subject);
  if (_options.positive_file!=NULL)
    free_2d_float(positivemask);
  
  free_2d_float(filtered);
  free_2d_float(segmented);
  free_2d_float(segsubject);
  free_2d_float(patchcount);

  free_3d_char(images);
  free_3d_char(masks);
  free_3d_char(means);
  free_3d_char(vars);
  
  free_meta(meta[2]);
  free_meta(meta[1]);
  free_meta(meta[0]);
  
  free(meta);
  
  cleanup_arguments(&_options);

  return STATUS_OK;
}

/* kate: indent-mode cstyle; indent-width 2; replace-tabs on; */
