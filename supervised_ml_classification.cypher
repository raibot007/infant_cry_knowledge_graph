//adding all the features in Audio Nodes
load csv with HEADERS from "https://raw.githubusercontent.com/raibot007/infant_cry_knowledge_graph/main/audio_features.csv" AS row
MERGE (a:Audio {id:row.file_id})
	SET a.filename = row.original_filename, a.dataset = row.dataset, a.class = row.class, a.duration = toFloat(row.original_duration), a.snr_estimate_db = toFloat(row.snr_estimate_db), a.dynamic_range_db = toFloat(row.dynamic_range_db), a.zero_crossing_rate = toFloat(row.zero_crossing_rate), a.gender = CASE WHEN row.gender IS NOT NULL THEN row.gender ELSE null END, a.age_months = CASE WHEN row.age_months IS NOT NULL THEN toInteger(row.age_months) ELSE null END, a.original_sample_rate = toInteger(row.original_sample_rate), a.target_sample_rate = toInteger(row.target_sample_rate), a.created_date = datetime(), a.rms_mean= toFloat(row.rms_mean), a.rms_std= toFloat(row.rms_std), a.zcr_mean= toFloat(row.zcr_mean), a.zcr_std= toFloat(row.zcr_std), a.amplitude_mean= toFloat(row.amplitude_mean), a.amplitude_std= toFloat(row.amplitude_std), a.amplitude_skewness= toFloat(row.amplitude_skewness), a.amplitude_kurtosis= toFloat(row.amplitude_kurtosis), a.amplitude_modulation_depth= toFloat(row.amplitude_modulation_depth), a.amplitude_dynamic_range= toFloat(row.dynamic_range), a.temporal_centroid= toFloat(row.temporal_centroid), a.autocorr_peak_value= toFloat(row.autocorr_peak_value), a.autocorr_peak_lag= toFloat(row.autocorr_peak_lag), a.silence_ratio= toFloat(row.silence_ratio), a.burst_density= toFloat(row.burst_density), a.spectral_centroid_mean= toFloat(row.spectral_centroid_mean), a.spectral_centroid_std= toFloat(row.spectral_centroid_std), a.spectral_bandwidth_mean= toFloat(row.spectral_bandwidth_mean), a.spectral_rolloff_mean= toFloat(row.spectral_rolloff_mean), a.spectral_flatness_mean= toFloat(row.spectral_flatness_mean), a.spectral_flux_mean= toFloat(row.spectral_flux_mean), a.spectral_slope_mean= toFloat(row.spectral_slope_mean), a.spectral_skewness_mean= toFloat(row.spectral_skewness_mean), a.spectral_kurtosis_mean= toFloat(row.spectral_kurtosis_mean), a.spectral_irregularity= toFloat(row.spectral_irregularity), a.mfcc1_mean= toFloat(row.mfcc_1_mean), a.mfcc1_std= toFloat(row.mfcc_1_std), a.mfcc2_mean= toFloat(row.mfcc_2_mean), a.mfcc2_std= toFloat(row.mfcc_2_std), a.mfcc3_mean= toFloat(row.mfcc_3_mean), a.mfcc3_std= toFloat(row.mfcc_3_std), a.mfcc4_mean= toFloat(row.mfcc_4_mean), a.mfcc4_std= toFloat(row.mfcc_4_std), a.mfcc5_mean= toFloat(row.mfcc_5_mean), a.mfcc5_std= toFloat(row.mfcc_5_std), a.mfcc6_mean= toFloat(row.mfcc_6_mean), a.mfcc6_std= toFloat(row.mfcc_6_std), a.mfcc7_mean= toFloat(row.mfcc_7_mean), a.mfcc7_std= toFloat(row.mfcc_7_std), a.mfcc8_mean= toFloat(row.mfcc_8_mean), a.mfcc8_std= toFloat(row.mfcc_8_std), a.mfcc9_mean= toFloat(row.mfcc_9_mean), a.mfcc9_std= toFloat(row.mfcc_9_std), a.mfcc10_mean= toFloat(row.mfcc_10_mean), a.mfcc10_std= toFloat(row.mfcc_10_std), a.mfcc11_mean= toFloat(row.mfcc_11_mean), a.mfcc11_std= toFloat(row.mfcc_11_std), a.mfcc12_mean= toFloat(row.mfcc_12_mean), a.mfcc12_std= toFloat(row.mfcc_12_std), a.mfcc13_mean= toFloat(row.mfcc_13_mean), a.mfcc13_std= toFloat(row.mfcc_13_std), a.f0_mean= toFloat(row.f0_mean), a.f0_std= toFloat(row.f0_std), a.harmonicity= toFloat(row.harmonicity), a.pitch_strength= toFloat(row.pitch_strength), a.loudness_mean= toFloat(row.loudness_mean), a.loudness_std= toFloat(row.loudness_std), a.sharpness_mean= toFloat(row.sharpness_mean), a.roughness= toFloat(row.roughness), a.tonality_coefficient= toFloat(row.tonality_coefficient), a.fluctuation_strength= toFloat(row.fluctuation_strength), a.C= toFloat(row.chroma_1), a.C_sharp= toFloat(row.chroma_2), a.D= toFloat(row.chroma_3), a.D_sharp= toFloat(row.chroma_4), a.E= toFloat(row.chroma_5), a.F= toFloat(row.chroma_6), a.F_sharp= toFloat(row.chroma_7), a.G= toFloat(row.chroma_8), a.G_sharp= toFloat(row.chroma_9), a.A= toFloat(row.chroma_10), a.A_sharp= toFloat(row.chroma_11), a.B= toFloat(row.chroma_12), a.spectral_contrast_sub_band= toFloat(row.spectral_contrast_sub_band), a.spectral_contrast_band_1= toFloat(row.spectral_contrast_band_1), a.spectral_contrast_band_2= toFloat(row.spectral_contrast_band_2), a.spectral_contrast_band_3= toFloat(row.spectral_contrast_band_3), a.spectral_contrast_band_4= toFloat(row.spectral_contrast_band_4), a.spectral_contrast_band_5= toFloat(row.spectral_contrast_band_5), a.spectral_contrast_band_6= toFloat(row.spectral_contrast_band_6), a.jitter= toFloat(row.jitter), a.shimmer= toFloat(row.shimmer), a.f0_range= toFloat(row.f0_range), a.f0_contour_slope= toFloat(row.f0_contour_slope), a.vocal_effort= toFloat(row.vocal_effort), a.f1_mean= toFloat(row.f1_mean), a.f2_mean= toFloat(row.f2_mean), a.f3_mean= toFloat(row.f3_mean)

// Create a lookup for classes
MATCH (n:Audio)
WITH DISTINCT n.class AS label
ORDER BY label
WITH collect(label) AS labels
UNWIND range(0, size(labels)-1) AS idx
MERGE (c:Class {name: labels[idx]})
SET c.id = idx

// Assign each Audio node its numeric classId
MATCH (n:Audio)
MATCH (c:Class {name: n.class})
SET n.classId = c.id

//GDS Graph projection
CALL gds.graph.project(
  'audio-ml-graph',
  {
    Audio: {
      properties: [
        'rms_mean','rms_std','zcr_mean','zcr_std',
        'amplitude_mean','amplitude_std','amplitude_skewness','amplitude_kurtosis',
        'amplitude_modulation_depth','amplitude_dynamic_range','temporal_centroid',
        'autocorr_peak_value','autocorr_peak_lag','silence_ratio','burst_density',
        'spectral_centroid_mean','spectral_centroid_std','spectral_bandwidth_mean',
        'spectral_rolloff_mean','spectral_flatness_mean','spectral_flux_mean',
        'spectral_slope_mean','spectral_skewness_mean','spectral_kurtosis_mean',
        'spectral_irregularity',
        'mfcc1_mean','mfcc1_std','mfcc2_mean','mfcc2_std','mfcc3_mean','mfcc3_std',
        'mfcc4_mean','mfcc4_std','mfcc5_mean','mfcc5_std','mfcc6_mean','mfcc6_std',
        'mfcc7_mean','mfcc7_std','mfcc8_mean','mfcc8_std','mfcc9_mean','mfcc9_std',
        'mfcc10_mean','mfcc10_std','mfcc11_mean','mfcc11_std','mfcc12_mean','mfcc12_std',
        'mfcc13_mean','mfcc13_std',
        'f0_mean','f0_std','harmonicity','pitch_strength','loudness_mean','loudness_std',
        'sharpness_mean','roughness','tonality_coefficient','fluctuation_strength',
        'C','C_sharp','D','D_sharp','E','F','F_sharp','G','G_sharp','A','A_sharp','B',
        'spectral_contrast_sub_band','spectral_contrast_band_1','spectral_contrast_band_2',
        'spectral_contrast_band_3','spectral_contrast_band_4','spectral_contrast_band_5',
        'spectral_contrast_band_6',
        'jitter','shimmer','f0_range','f0_contour_slope','vocal_effort',
        'f1_mean','f2_mean','f3_mean',
		'classId'
      ]
    }
  },
  '*'
)


//ML pipeline creation
CALL gds.beta.pipeline.nodeClassification.create('audio-pipeline')

// Select which features the pipeline should use
CALL gds.beta.pipeline.nodeClassification.selectFeatures(
  'audio-pipeline',
  [
    'rms_mean','rms_std','zcr_mean','zcr_std',
    'amplitude_mean','amplitude_std','amplitude_skewness','amplitude_kurtosis',
    'amplitude_modulation_depth','amplitude_dynamic_range','temporal_centroid',
    'autocorr_peak_value','autocorr_peak_lag','silence_ratio','burst_density',
    'spectral_centroid_mean','spectral_centroid_std','spectral_bandwidth_mean',
    'spectral_rolloff_mean','spectral_flatness_mean','spectral_flux_mean',
    'spectral_slope_mean','spectral_skewness_mean','spectral_kurtosis_mean',
    'spectral_irregularity',
    'mfcc1_mean','mfcc1_std','mfcc2_mean','mfcc2_std','mfcc3_mean','mfcc3_std',
    'mfcc4_mean','mfcc4_std','mfcc5_mean','mfcc5_std','mfcc6_mean','mfcc6_std',
    'mfcc7_mean','mfcc7_std','mfcc8_mean','mfcc8_std','mfcc9_mean','mfcc9_std',
    'mfcc10_mean','mfcc10_std','mfcc11_mean','mfcc11_std','mfcc12_mean','mfcc12_std',
    'mfcc13_mean','mfcc13_std',
    'f0_mean','f0_std','harmonicity','pitch_strength','loudness_mean','loudness_std',
    'sharpness_mean','roughness','tonality_coefficient','fluctuation_strength',
    'C','C_sharp','D','D_sharp','E','F','F_sharp','G','G_sharp','A','A_sharp','B',
    'spectral_contrast_sub_band','spectral_contrast_band_1','spectral_contrast_band_2',
    'spectral_contrast_band_3','spectral_contrast_band_4','spectral_contrast_band_5',
    'spectral_contrast_band_6',
    'jitter','shimmer','f0_range','f0_contour_slope','vocal_effort',
    'f1_mean','f2_mean','f3_mean'
  ]
)


// Add model candidates: logistic regression, random forest
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('audio-pipeline')
CALL gds.beta.pipeline.nodeClassification.addRandomForest(
  'audio-pipeline',
  {
    maxDepth: 10,
    minSplitSize: 2,
    numberOfDecisionTrees: 50,   // instead of numberOfTrees
    numberOfSamplesRatio: 1.0    // fraction of training data per tree
  }
)


// Train the pipeline 
CALL gds.beta.pipeline.nodeClassification.train(
  'audio-ml-graph',
  {
    pipeline: 'audio-pipeline',
    targetNodeLabels: ['Audio'],         // which nodes to classify
    targetProperty: 'classId',             // property on Audio nodes (must exist)
    modelName: 'audio-classifier',
    testFraction: 0.2,
    validationFolds: 5,
    randomSeed: 42,
    metrics: ['F1_MACRO', 'ACCURACY']
  }
)
YIELD modelInfo, modelSelectionStats
RETURN modelInfo, modelSelectionStats

