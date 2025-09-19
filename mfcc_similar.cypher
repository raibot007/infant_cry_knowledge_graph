call apoc.periodic.iterate('
MATCH (a1:Audio)-[:HAS_MFCC]->(m1:MFCC)
MATCH (a2:Audio)-[:HAS_MFCC]->(m2:MFCC)
WHERE a1.dataset <> a2.dataset
WITH id(a1) as a1, id(a2) as a2, id(m1) as m1, id(m2) as m2,
     [m1.mfcc1_mean, m1.mfcc2_mean, m1.mfcc3_mean, m1.mfcc4_mean, m1.mfcc5_mean,
      m1.mfcc6_mean, m1.mfcc7_mean, m1.mfcc8_mean, m1.mfcc9_mean, m1.mfcc10_mean,
      m1.mfcc11_mean, m1.mfcc12_mean, m1.mfcc13_mean] as mfcc1_vector,
     [m2.mfcc1_mean, m2.mfcc2_mean, m2.mfcc3_mean, m2.mfcc4_mean, m2.mfcc5_mean,
      m2.mfcc6_mean, m2.mfcc7_mean, m2.mfcc8_mean, m2.mfcc9_mean, m2.mfcc10_mean,
      m2.mfcc11_mean, m2.mfcc12_mean, m2.mfcc13_mean] as mfcc2_vector
WITH a1, a2, gds.similarity.cosine(mfcc1_vector, mfcc2_vector) as mfcc_similarity
WHERE mfcc_similarity > 0.95
WITH a1, a2, mfcc_similarity
WHERE a1 < a2
RETURN a1, a2, mfcc_similarity',
'WITH a1, a2, mfcc_similarity
MATCH (a:Audio) where id(a)=a1
MATCH (b:Audio) where id(b)=a2
MERGE (a)-[:MFCC_SIMILAR {similarity: mfcc_similarity, method: "cosine"}]->(b)',
{batchSize:10000, parallel:false})
yield total, batches, failedBatches, errorMessages
RETURN total, batches, failedBatches, errorMessages