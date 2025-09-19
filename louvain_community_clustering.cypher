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

//louvain community detection
CALL gds.louvain.write('audio-feature-similarity', {
    writeProperty: 'acoustic_community',
    relationshipWeightProperty: 'similarity',
    tolerance: 0.001,
    maxIterations: 20,
    includeIntermediateCommunities: true
})
YIELD communityCount, modularity, ranLevels

//fetching community clusters
MATCH (a:Audio)
WITH a.acoustic_community as community, 
     collect(a) as audio_nodes
WHERE size(audio_nodes) > 2
RETURN community,
       size(audio_nodes) as community_size,
       size([a in audio_nodes WHERE a.dataset = "DonateACry"]) as donateacry_count,
       size([a in audio_nodes WHERE a.dataset = "CRIED"]) as cried_count,
       size([a in audio_nodes WHERE a.dataset = "BabyChillanto"]) as babychillanto_count,
       apoc.coll.toSet([n in audio_nodes | n.class]) AS classes_in_community
ORDER BY community_size DESC
