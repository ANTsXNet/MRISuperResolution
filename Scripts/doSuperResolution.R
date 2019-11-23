library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )

# k_clear_session()
# threads <- 1L
# config <- tf$compat$v1$ConfigProto( intra_op_parallelism_threads = threads,
#                                     inter_op_parallelism_threads = threads )
# session <- tf$compat$v1$Session( config = config )
# k_set_session( session )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) < 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript doSuperResolution.R inputFile outputFile\n" )
  stop( helpMessage )
  } else {
  inputFile <- args[1]
  outputFile <- args[2]
  }

startTimeTotal <- Sys.time()

cat( "Reading ", inputFile )
startTime <- Sys.time()
inputImage <- antsImageRead( inputFile )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

dimension <- length( dim( inputImage ) )

inputImageList <- list()
if( dimension == 4 )
  {
  inputImageList <- splitNDImageToList( inputImage )
  } else if( dimension == 2 ) {
  stop( "Error:  model for 3-D or 4-D images only." )
  } else if( dimension == 3 ) {
  inputImageList[[1]] <- inputImage
  }

model <- createDeepBackProjectionNetworkModel3D( c( dim( inputImageList[[1]] ),  1 ),
  numberOfOutputs = 1, numberOfBaseFilters = 64,
  numberOfFeatureFilters = 256, numberOfBackProjectionStages = 7,
  convolutionKernelSize = c( 3, 3, 3 ),
  strides = c( 2, 2, 2 ), numberOfLossFunctions = 1 )

cat( "Loading weights file" )
startTime <- Sys.time()
weightsFileName <- paste0( getwd(), "/mriSuperResolutionWeights.h5" )
if( ! file.exists( weightsFileName ) )
  {
  weightsFileName <- getPretrainedNetwork( "mriSuperResolution", weightsFileName )
  }
model$load_weights( weightsFileName )

numberOfImageVolumes <- length( inputImageList )

outputImageList <- list()
for( i in seq_len( numberOfImageVolumes ) )
  {
  cat( "Applying super resolution to image", i, "of", numberOfImageVolumes, "\n" )

  startTime <- Sys.time()
  inputImage <- iMath( inputImageList[[i]], "TruncateIntensity", 0.0001, 0.995 )
  outputSR <- applySuperResolutionModelToImage( inputImage, model, targetRange = c( 127.5, -127.5 ) )
  inputImageResampled <- resampleImageToTarget( inputImage, outputSR )
  outputImageList[[i]] <- regressionMatchImage( outputSR, inputImageResampled, polyOrder = 2 )

  endTime <- Sys.time()
  elapsedTime <- endTime - startTime
  cat( "   (elapsed time:", elapsedTime, "seconds)\n" )
  }

cat( "Writing output image.\n" )
if( numberOfImageVolumes == 1 )
  {
  antsImageWrite( outputImageList[[1]], outputFile )
  } else {
  outputImage <- mergeListToNDImage( inputImage, outputImageList )
  antsImageWrite( outputImage, outputFile )
  }

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "  (Total elapsed time:", elapsedTimeTotal, "seconds)\n" )
