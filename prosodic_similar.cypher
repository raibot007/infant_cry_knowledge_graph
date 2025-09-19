call apoc.periodic.iterate('
MATCH (a1:Audio)-[:HAS_PROSODIC_FEATURES]->(p1:Prosodic)
MATCH (a2:Audio)-[:HAS_PROSODIC_FEATURES]->(p2:Prosodic)
WHERE a1.dataset <> a2.dataset
WITH id(a1) as a1, id(a2) as a2, id(p1) as p1, id(p2) as p2,
     [p1.f0_mean, p1.f0_std, p1.harmonicity, p1.pitch_strength] as prosodic1_vector,
     [p2.f0_mean, p2.f0_std, p2.harmonicity, p2.pitch_strength] as prosodic2_vector
WITH a1, a2, gds.similarity.cosine(prosodic1_vector, prosodic2_vector) as prosodic_similarity
WHERE prosodic_similarity > 0.95
WITH a1, a2, prosodic_similarity
WHERE a1 < a2
RETURN a1, a2, prosodic_similarity',
'WITH a1, a2, prosodic_similarity
MATCH (a:Audio) where id(a)=a1
MATCH (b:Audio) where id(b)=a2
MERGE (a)-[:PROSODIC_SIMILAR {similarity: prosodic_similarity, method: "cosine"}]->(b)',
{batchSize:10000, parallel:false})
yield total, batches, failedBatches, errorMessages
RETURN total, batches, failedBatches, errorMessages