/*  nlmsegFuzzy.c
 * 
 *  Copyright 2011  Simon Fristed Eskildsen, Vladimir Fonov,
 *   	      	    Pierrick Coupé, Jose V. Manjon
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


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <strings.h>
#include <string.h>
#include <float.h>
#include "nlmseg.h"

#define MINCOUNT 100


#ifdef MT_USE_OPENMP
    #include <omp.h>
#else
    #define omp_get_num_threads() 1
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
#endif


float nlmsegSparse4D(float *subject, float *imagedata, 
                     float *maskdata, float *meandata, float *vardata, 
                     float *mask, 
                     int sizepatch, int searcharea, float beta, float threshold, 
                     int dims[3],  int librarysize, float *SegSubject, float *PatchCount)
{    
  float *MeansSubj, *VarsSubj, *localmask;
  int i,v,f,ndim;
  float min,max;
  int patch_volume;
  int patch_center_voxel;
  
  int sadims,volumesize,index;
  int mincount=MINCOUNT;
  int notfinished;
  double minidist;
  double epsi = 0.0001;
  time_t time1,time2;
  
  /*per thread temp storage*/
  
  float  **_PatchImg  =(float **) malloc(omp_get_max_threads()*sizeof(float*));
  float  **_PatchMask =(float **) malloc(omp_get_max_threads()*sizeof(float*));
  float  **_PatchTemp =(float **) malloc(omp_get_max_threads()*sizeof(float*));
  float  **_PatchDistance=(float **) malloc(omp_get_max_threads()*sizeof(float*));
  

  fprintf(stderr,"Patch size: %d\nSearch area: %d\nBeta: %f\nThreshold: %f\nSelection: %d\n",sizepatch,searcharea,beta,threshold,librarysize);
  
  ndim = 3;
  volumesize=dims[0]*dims[1]*dims[2];
  
  /*Patch radius*/
  f = sizepatch;
  /*volume of patch*/
  patch_volume=(2*f+1)*(2*f+1)*(2*f+1);
  /*index of patch central voxel*/
  patch_center_voxel=f+f*(2*f+1)+f*(2*f+1)*(2*f+1);

  /*Search Area radius*/
  v = searcharea;
  
  sadims = pow(2*v+1,ndim);
  sadims = sadims * librarysize;

  /* allocate memory for multithreading operation*/
  for(i=0;i<omp_get_max_threads();i++)
  {
    _PatchTemp[i]=(float*) malloc( (2*f+1)*(2*f+1)*(2*f+1)*sizeof(float) );
  }
  
  
  MeansSubj = (float *)calloc(volumesize,sizeof(*MeansSubj));
  VarsSubj =  (float *)calloc(volumesize,sizeof(*VarsSubj));
  localmask = (float *)calloc(volumesize,sizeof(*localmask));
  
  memmove(localmask,mask,volumesize*sizeof(*localmask));
  
  fprintf(stderr,"Dimensions: %d %d %d\n",dims[0],dims[1],dims[2]);
  
  fprintf(stderr,"Computing first moment image...");
  time1=time(NULL);
  ComputeFirstMoment(subject, MeansSubj, dims, f, &min, &max);
  time2=time(NULL);
  fprintf(stderr,"done (%d sec)\nComputing second moment image...",(int)(time2-time1));
  ComputeSecondMoment(subject, MeansSubj, VarsSubj, dims, f, &min, &max);
  fprintf(stderr,"done");
  time1=time(NULL);
  
  do {
    
    fprintf(stderr," (%d sec)\nSegmenting      ",(int)(time1-time2));
    time2=time(NULL);
    notfinished=0;
    
    for(i=0;i<omp_get_max_threads();i++)
    {
      _PatchImg[i]  =(float*) malloc( patch_volume*sizeof(float)*sadims );
      _PatchMask[i] =(float*) malloc( patch_volume*sizeof(float)*sadims );
      _PatchDistance[i]=(float*) malloc( sizeof(float)*sadims );
    }
    
    #pragma omp parallel for shared(_PatchImg,_PatchTemp,_PatchDistance,_PatchMask) reduction(+:notfinished)
    for(i=0;i<dims[0];i++)
    { /*start parallel section*/
      int j,k;
      
      /*use thread-specific temp memory*/
      float * PatchImg= _PatchImg[omp_get_thread_num()];
      float * PatchMask=_PatchMask[omp_get_thread_num()];;
      float * PatchTemp=_PatchTemp[omp_get_thread_num()];
      float * PatchDistance= _PatchDistance[omp_get_thread_num()];
      
      if( omp_get_thread_num()==0 )
        fprintf(stderr,"\b\b\b\b\b\b\b\b\b%3d / %3d", i*omp_get_num_threads()+1, dims[0]);
      
      for(j=0;j<dims[1];j++)
      {
        for(k=0;k<dims[2];k++)
        {
          int index=i*(dims[2]*dims[1])+(j*dims[2])+k;
          
          /* mask check */
          if ( localmask[index]  > 0 )
          {
            int ii,jj,kk;
            float proba=0.;
            float totalweight=0;
            int   count = 0;
            
            float TMean,TVar;
            
            ExtractPatch(subject, PatchTemp, i, j, k, f, dims[0], dims[1], dims[2]);
            
            TMean = MeansSubj[index];
            TVar =  VarsSubj [index];
            
            /*TODO:apply patch-normalization using Mean and var*/
            
            /* go through the search space  */
            for(ii=-v;ii<=v;ii++)
            {
              for(jj=-v;jj<=v;jj++)
              {
                for(kk=-v;kk<=v;kk++)
                {
                  int ni,nj,nk;
                  ni=i+ii;
                  nj=j+jj;
                  nk=k+kk;
                  
                  if( (ni>=0) && (nj>=0) && (nk>=0) && 
                      (ni<dims[0]) && (nj<dims[1]) && (nk<dims[2]) )
                  {
                    int t;
                    for(t=0;t<librarysize;t++)
                    {
                      int index2 = t*volumesize+ni*(dims[2]*dims[1])+(nj*dims[2])+nk;
                      
                      float Mean = meandata[index2];
                      float Var =  vardata [index2];
                      
                      /*Similar Luminance and contrast -> Cf Wang TIP 2004*/
                      float th = ((2 * Mean * TMean + epsi) / ( Mean*Mean + TMean*TMean + epsi))  *
                                 ((2 * sqrt(Var) * sqrt(TVar) + epsi) / (Var + TVar + epsi));

                      if(th > threshold)
                      {
                        ExtractPatch4D(imagedata, PatchImg+ count*patch_volume ,ni,nj,nk, t, f, dims[0], dims[1], dims[2]);
                        ExtractPatch4D(maskdata,  PatchMask+count*patch_volume ,ni,nj,nk, t, f, dims[0], dims[1], dims[2]);
                        
                        /*TODO:apply patch-normalization using Mean and Var here?*/
                        
                        count ++;
                        
                      }
                    }
                  }
                }
              }
            }
            
            /* require a minimum number of selected patches  */
            if (count >= mincount) {
                float average=0.0;
                int realcount=0;
                int p;
                
                /*TODO: replace with sparse code*/
                float minidist = FLT_MAX; /*FLT_MAX;*/
                for(p=0;p<count;p++)
                {
                    /*calculate distances*/
                    float d =  SSDPatch(PatchImg+p*patch_volume, PatchTemp, f);
                    if (d < minidist) minidist = d;
                    PatchDistance[p]=d;
                }
                
                if ( minidist<=epsi ) minidist = epsi; /*to avoid division by zero*/
                
                for(p=0;p<count;p++)
                {
                    float w = exp(- ((PatchDistance[p])/(beta*(minidist)) ) ); /*The smoothing parameter is the minimal distance*/
                    
                    if (w>0.0)  
                    {
                        average+= PatchMask[p*patch_volume+patch_center_voxel]*w;
                        totalweight += w;
                        realcount++;
                    }
                } 
                
                /* We compute the probability */
                proba = average / totalweight;
                
                SegSubject[index] = proba;
                PatchCount[index] = realcount;
            } else {
              /* Not enough similar patches */
              notfinished+=1;
              SegSubject[index] = -1;
            }
          
          }// mask check
          
        } // for k
      } // for j
      
      /*Freeing per-thread data*/
      /*end of parallel section*/
    } // for i
    
    time1=time(NULL);
    
    if ( notfinished>0 ) {
      /*relax preselection criteria*/
      int count;
      threshold=threshold*0.95;
      mincount=mincount*0.95;
      v=v+1;
      count=0;
      
      #pragma omp parallel for reduction(+:count)
      for(i=0;i<dims[0];i++)
      {
        int j,k;
        for(j=0;j<dims[1];j++)
        {
          for(k=0;k<dims[2];k++)
          {
            int index=i*(dims[2]*dims[1])+(j*dims[2])+k;
            
            if ( SegSubject[index]<0 ){
              localmask[index] = 1;
              count++;
            } else {
              localmask[index] = 0;
            }
          }
        }
      }
      
      fprintf(stderr," (redoing %d voxels) t=%f, min=%d ",count, threshold, mincount);
      sadims = pow( 2*v+1,ndim);
      sadims = sadims * librarysize;
    }
    
    for(i=0;i<omp_get_max_threads();i++)
    {
        free(_PatchImg[i]);
        free(_PatchDistance[i]);
    }

  } while (notfinished);

  for(i=0;i<omp_get_max_threads();i++)
  {
    free(_PatchTemp[i]);
    
  }
  
  free(_PatchImg);
  free(_PatchTemp);
  free(_PatchDistance);
  
  
  fprintf(stderr," done (%d sec, t=%f, min=%d)\n",(int)(time1-time2), threshold, mincount);
  
  free(MeansSubj);
  free(VarsSubj);
  free(localmask);
  
  return max;
}


