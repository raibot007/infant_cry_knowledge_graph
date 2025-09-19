//GDS Graph Projection
CALL gds.graph.project(
    'audio-feature-similarity',
    'Audio',  // Only Audio nodes for community detection
    {
        MFCC_SIMILAR: {
            orientation: 'UNDIRECTED',
            properties: 'similarity'
        },
        PROSODIC_SIMILAR: {
            orientation: 'UNDIRECTED', 
            properties: 'similarity'
        }
    }
)

//node2vec embeddings
CALL gds.node2vec.write('audio-feature-similarity', {
    writeProperty: 'node2vec_embedding',
    embeddingDimension: 128,
    walkLength: 10,
    walksPerNode: 20
})

//class_embedding creation
MATCH (c:Class)-[:HAS_AUDIO]->(a:Audio)
WITH c, collect(a.node2vec_embedding) as vectors
WITH c, range(0, size(vectors[0])-1) AS idxs, vectors
UNWIND idxs AS i
WITH c, collect( reduce(sum=0, x IN vectors | sum + x[i]) * 1.0 / size(vectors) ) as class_embedding
SET c.class_embedding = class_embedding

//cross dataset similar classes
MATCH (c1:Class), (c2:Class)
WHERE c1.dataset <> c2.dataset
WITH c1, c2, gds.similarity.cosine(c1.class_embedding, c2.class_embedding) AS embedding_similarity
// Group by dataset of c2
WITH c1, c2.dataset AS dataset, collect({c2:c2, score:embedding_similarity}) AS sims
WITH c1, dataset, 
     reduce(maxScore = 0.0, s IN sims | CASE WHEN s.score > maxScore THEN s.score ELSE maxScore END) AS maxSim,
     sims
// Pick the best in each dataset
UNWIND sims AS s
WITH c1, dataset, s
WHERE s.score = reduce(maxScore = 0.0, t IN sims | CASE WHEN t.score > maxScore THEN t.score ELSE maxScore END)
// Create the relationship
WITH c1, s.score as score, s.c2 as c2
MERGE (c1)-[:SEMANTICALLY_SIMILAR {
    similarity: score,
    method: "node2vec_cosine",
    discovered_by: "GDS"
}]->(c2)
