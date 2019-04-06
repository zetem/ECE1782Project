/*
link to YUV 
http://trace.eas.asu.edu/yuv/index.html
*/
#include <math.h>  // for abs
#include <stdio.h>
#include <stdint.h> // for uint8_t
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>

//using namespace std;
#define PIXEL uint8_t

#define H 288 // height of each frame
#define W 352 // width of each frame
#define NF 10 // number of frames
#define NRF 3 // number of refernce frames
#define BS  2 // block_size
#define SR  2 // search_radius


//because using c not c++
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// time stamp function in seconds 
double getTimeStamp()
{     
    struct timeval  tv ;     
    gettimeofday( &tv, NULL ) ;     
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
} 


//bx by are cordinations for block
// r is search Radius
// i is block size 
/* CURRENLTY NO USE
int Calculate_SAD(PIXEL *currunt_block, int cur_y, int cur_x, PIXEL *reference_block, int ref_y, int ref_x, int block_size)
{
    int total_diff = 0;
    for(int j =0; j < block_size; j++)
    {
        for (int i=0; i < block_size; i++)
        {
            total_diff += abs(currunt_block[(cur_y+j)*block_size+cur_x+i] - reference_frame[ref_y+j*bo +ref_x+i])
        }
    }
    return total_diff;
}
*/

int generate_mv_for_block(PIXEL *currunt_frame, PIXEL *reference_frames, int y, int x, int r, int i, int *motion_vector_y, int *motion_vector_x)    
{    
    int lowest_SAD = 256 * i * i;// variable storing the SAD value
    int block_size = BS;
    // (y,x) the pixel location of the top-left corner of the block.
    
    //Search for the best matching block in the reference frame.
    //The search processes only the block is within the reference frame (not out of boundary).
    for (int ref_index = 0; ref_index < NRF; ref_index++)
    {        
        for (int search_x_radius = MAX(x-r,0); search_x_radius <= MIN(x+r,W-i); search_x_radius++){
            
            for (int search_y_radius = MAX(y-r,0); search_y_radius <= MIN(y+r,H-i) ; search_y_radius++){                
                
                //Calculate SAD of this block with the input block.
                //int SAD =  Calculate_SAD(currunt_frame,y,x,referenceFrame[ref_index],search_y_radius,search_x_radius);
                int SAD = 0;
                for(int j =0; j < block_size; j++)
                {
                    for (int i=0; i < block_size; i++)
                    {
                        SAD += abs(currunt_frame[(y+j)*W+x+i] - reference_frames[(search_y_radius+j)*W +search_x_radius+i]);
                    }
                }
                //If this block is better in SAD...
                if (lowest_SAD > SAD){
                    lowest_SAD = SAD; // Update SAD.
                    (*motion_vector_x) = search_x_radius - x;
                    (*motion_vector_y) = search_y_radius - y; //Update motion vector.
                }
                //If there is a tie in SAD keep last change
            }
        }
    }
    return 1;
}


/*
void get_Y(FILE *fid, PIXEL *luma, int comps){
    PIXEL temp;
    for (int f=0; f<NRF; f++) {
        for (int kk=0; kk<comps; kk++){
            for (int h=0; h<H; h++){
                for (int w=0; w<W; w++){
                    int idx = f*W*H*3/2 +h*W +w;
                    if (kk==0){
                        fread (&luma[idx],sizeof(char),1,fid);
                    }
                    else if ((w%2==0 && h%2==0) || kk==0){
                        fread (&temp,sizeof(char),1,fid);
                    }
                }
            }
        }
    }
}*/

void read_luma(FILE *fid, PIXEL *luma){
    for (int f=0; f<NRF; f++) {
        fread (luma,1,W*H,fid);
        fseek (fid,W*H/2,SEEK_CUR); //seek cb and cr
    }
}

int main()
{
    // take of available refrence frames
    int block_size = BS; //min can be 2 , otherwise change malloc
    int search_radius = SR; 
    //motion_vector[H][W][2];
    // CIF format
    int height = H; int width = W;
    int number_frames = NF;
    ////////////////////////////////////
    FILE    *fid_in         = fopen("akiyo_cif.yuv","rb");
    FILE    *fid_out        = fopen("akiyo_qcif_constructed.y","wb");
    PIXEL   *luma           = (PIXEL *) malloc(height*width*number_frames*sizeof(PIXEL)); if (luma          == NULL) fprintf(stderr, "Bad malloc on luma           \n");
    PIXEL   *predicted      = (PIXEL *) malloc(height*width*number_frames*sizeof(PIXEL)); if (predicted     == NULL) fprintf(stderr, "Bad malloc on predicted      \n");
    PIXEL   *reconstructed  = (PIXEL *) malloc(height*width*number_frames*sizeof(PIXEL)); if (reconstructed == NULL) fprintf(stderr, "Bad malloc on reconstructed  \n");
    PIXEL   *motion_vector  = (PIXEL *) malloc(height*width*number_frames*sizeof(PIXEL)); if (motion_vector == NULL) fprintf(stderr, "Bad malloc on motion_vector  \n");
    PIXEL   *Res_orig       = (PIXEL *) malloc(height*width*number_frames*sizeof(PIXEL)); if (Res_orig      == NULL) fprintf(stderr, "Bad malloc on Res_orig       \n");
    PIXEL   *currunt_frame  = (PIXEL *) malloc(height*width              *sizeof(PIXEL)); if (currunt_frame == NULL) fprintf(stderr, "Bad malloc on currunt_frame  \n");
    PIXEL   *reference_frames= (PIXEL*) malloc(height*width*NRF          *sizeof(PIXEL)); if (reference_frames == NULL) fprintf(stderr, "Bad malloc on reference_frames  \n");
    read_luma(fid_in,luma);
    fclose(fid_in);
    double timeStampA = getTimeStamp() ; 
    for (int f =NRF; f < number_frames; f++)
    {
        memcpy(currunt_frame   ,&luma[height*width*f]      ,height*width    *sizeof(PIXEL));
        memcpy(reference_frames,&luma[height*width*(f-NRF)],height*width*NRF*sizeof(PIXEL));
        for (int y = 0; y < H; y+=block_size)
        {
            for (int x = 0; x < W; x+=block_size)
            {
                int mvy = 0;
                int mvx = 0;
                generate_mv_for_block(currunt_frame, reference_frames, y, x, search_radius, block_size, &mvy, &mvx);
                motion_vector[(y*W + x)*2+0] = mvy;
                motion_vector[(y*W + x)*2+1] = mvx;   
            }
        }
    }
    double timeStampB= getTimeStamp() ;
    printf("total CPU time = %.6f\n", timeStampB - timeStampA); 
}