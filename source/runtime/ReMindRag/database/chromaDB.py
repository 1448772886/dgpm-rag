from ..llms import AgentBase
from ..embeddings import EmbeddingBase
from ..chunking import ChunkerBase

from .data_extract import handle_file, handle_file_folder, handle_content
from ..utils.math_functions import edge_weight_coefficient
from ..utils.logger import setup_logger


import chromadb
import os
import hashlib

from typing import List, Dict, Any, Tuple
import numpy as np
import math
import logging
from transformers import AutoTokenizer


class ChromaDBManager:
    def __init__(self, 
                 logger_level,
                 chunk_agent:AgentBase, 
                 embedding:EmbeddingBase, 
                 chunker:ChunkerBase, 
                 synonym_threshold, 
                 edge_weight_alpha, 
                 strong_connection_threshold,
                 chromadp_pth,
                 tokenizer:AutoTokenizer,
                 log_path = None):
        self.logger = setup_logger("Database", logger_level, log_path)

        self.client = chromadb.PersistentClient(path = chromadp_pth)

        self.entity_collection = self.client.get_or_create_collection(name="entity")
        self.chunk_collection = self.client.get_or_create_collection(name="chunk")
        self.relation_collection = self.client.get_or_create_collection(name="relation")
        self.connection_collection = self.client.get_or_create_collection(name="connection")

        self.chunk_agent = chunk_agent
        self.chunker = chunker
        self.embedding = embedding
        self.hidden_size = embedding.get_hidden_state_size()

        self.max_chunk_id = self._get_max_chunk_id()
        self.synonym_threshold = synonym_threshold
        self.edge_weight_alpha = edge_weight_alpha
        self.strong_connection_threshold = strong_connection_threshold

        self.tokenizer = tokenizer

        self.dfs_entity = []
        self.dfs_chunk = []
        self.dfs_edge = []
        self.chromadp_pth = chromadp_pth  # (local addition) used by the extraction checkpoint
    
    def _get_max_chunk_id(self) -> int:
        chunks = self.chunk_collection.get()
        if not chunks["ids"]:
            return 0
        return max(int(chunk_id) for chunk_id in chunks["ids"])
    
    def add_file_data(self, file_pth, language, encoding):
        extracted_data = handle_file(self.logger, self.chunk_agent, self.chunker, file_pth, language, encoding)
        self.add_extracted_data(extracted_data)

    def add_folder_data(self, folder_file_pth, language, encoding):
        extracted_data = handle_file_folder(self.logger, self.chunk_agent, self.chunker, folder_file_pth, language, encoding)
        self.add_extracted_data(extracted_data)

    def add_content(self, content, language):
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
        checkpoint = os.path.join(self.chromadp_pth, f"extraction_checkpoint_{content_hash}.json")
        extracted_data = handle_content(self.logger, content, self.chunk_agent, self.chunker, language, checkpoint_path=checkpoint)
        self.add_extracted_data(extracted_data)


    def add_extracted_data(self, extracted_data):
        synonym_mappings = {}
        chunk_id_list = []
        for data in extracted_data:
            chunk_id = self.add_chunk(data["chunk"])
            chunk_id_list.append(chunk_id)
            
            for entity in data["entity"]:
                result = self.add_entity(entity, chunk_id, data["chunk"]["title"])
                if result:
                    original, replacement = result
                    synonym_mappings[original] = replacement
            
            for relation in data["relation"]:
                if relation[0] in synonym_mappings:
                    relation[0] = synonym_mappings[relation[0]]
                if relation[2] in synonym_mappings:
                    relation[2] = synonym_mappings[relation[2]]
                
                self.add_relation(relation)

        for i in range(len(chunk_id_list)-1):
            anchor_id1 = f"anchor-{chunk_id_list[i]}"
            anchor_id2 = f"anchor-{chunk_id_list[i+1]}"
            self.relation_collection.add(
                ids=[f"{anchor_id1}_{anchor_id2}"],
                metadatas=[{
                    "type": "relation",
                    "subject_entity_id": anchor_id1,
                    "object_entity_id": anchor_id2
                }],
                documents=[f"The following part of chunk:{chunk_id_list[i]} is chunk:{chunk_id_list[i+1]}."],
                embeddings=[np.zeros(self.hidden_size, dtype=np.float32)]
                # embeddings=[self.embedding.sentence_embedding(relation_text)]
            )
    

    def add_entity(self, entity, chunk_id, chunk_title):
        similar_entity = self.query_similar_entity(entity)
        anchor_id = f"anchor-{chunk_id}"
        
        if similar_entity:
            self.relation_collection.add(
                ids=[f"{similar_entity}_{anchor_id}"],
                metadatas=[{
                    "type": "relation",
                    "subject_entity_id": similar_entity,
                    "object_entity_id": anchor_id
                }],
                documents=[f"relation of entity:{similar_entity} to entity:{anchor_id}"],
                embeddings=[np.zeros(self.hidden_size, dtype=np.float32)]
            )
            self.logger.debug(f"Trans Entity:{entity} to Similar Entity:{similar_entity}.")
            return (entity, similar_entity)
        
        self.entity_collection.add(
            ids=[entity],
            metadatas=[{"type": "entity"}],
            documents=[entity],
            embeddings=[self.embedding.sentence_embedding(entity)]
        )

        self.relation_collection.add(
            ids=[f"{entity}_{anchor_id}"],
            metadatas=[{
                "type": "relation",
                "subject_entity_id": entity,
                "object_entity_id": anchor_id
            }],
            documents=[f"relation of entity:{entity} to entity:{anchor_id}"],
            embeddings=[np.zeros(self.hidden_size, dtype=np.float32)]
        )
        
        return None
    
    def add_missing_entity(self, entity):
        similar_entity = self.query_similar_entity(entity)
        
        if similar_entity:
            self.logger.debug(f"Trans Entity:{entity} to Similar Entity:{similar_entity} While Adding Missing Entity.")
            return (entity, similar_entity)
        else:
            self.entity_collection.add(
                ids=[entity],
                metadatas=[{"type": "entity"}],
                documents=[entity],
                embeddings=[self.embedding.sentence_embedding(entity)]
            )

    def add_chunk(self, chunk):
        self.max_chunk_id += 1
        chunk_id = str(self.max_chunk_id)
        anchor_id = f"anchor-{chunk_id}"

        self.entity_collection.add(
            ids=[anchor_id],
            metadatas=[{"type": "entity"}],
            documents=[chunk["title"]],
            # embeddings=[self.embedding.sentence_embedding(chunk["title"])]
            embeddings=[self.embedding.sentence_embedding(chunk["content"])]
        )

        chunk_token_count = len(self.tokenizer.tokenize(chunk["content"]))
        
        self.chunk_collection.add(
            ids=[chunk_id],
            metadatas=[{"type": "chunk","title":chunk["title"],"tokens":chunk_token_count}],
            documents=[chunk["content"]],
            embeddings=[self.embedding.sentence_embedding(chunk["content"])]
        )

        self.connection_collection.add(
            ids=[f"{anchor_id}_{chunk_id}"],
            embeddings = [np.zeros(self.hidden_size, dtype=np.float32)],
            metadatas=[{"entity_id": anchor_id, "chunk_id": chunk_id}],
            documents = [chunk["title"]]
        )
        
        return chunk_id

    def get_entity_chunk_area(self, entity_id: str):
        """Return chunk ids directly associated with an entity seed.

        This is deliberately a one-hop index lookup used only for diverse seed
        selection. It does not run graph traversal or apply query-dependent edge
        thresholds.
        """
        chunks = set()
        entity_id = str(entity_id)

        if entity_id.startswith("anchor-"):
            chunks.add(entity_id[len("anchor-"):])

        connection = self.connection_collection.get(
            where={"entity_id": entity_id}, include=["metadatas"])
        for metadata in connection.get("metadatas", []):
            chunk_id = metadata.get("chunk_id")
            if chunk_id is not None:
                chunks.add(str(chunk_id))

        relations = self.relation_collection.get(
            where={"$or": [
                {"subject_entity_id": entity_id},
                {"object_entity_id": entity_id},
            ]}, include=["metadatas"])
        for metadata in relations.get("metadatas", []):
            for key in ("subject_entity_id", "object_entity_id"):
                other = str(metadata.get(key, ""))
                if other.startswith("anchor-"):
                    chunks.add(other[len("anchor-"):])

        return chunks
    


    def add_relation(self, relation):
        subject_entity = relation[0]
        relation_text = relation[1]
        object_entity = relation[2]

        if subject_entity == object_entity:
            self.logger.debug(f"Self-Reference Relation: {relation}")
            return
        
        check_entity_set = self.entity_collection.get(ids = [subject_entity,object_entity])
        if len(check_entity_set["ids"]) == 0:
            self.logger.debug(f"Unknown Entity Found While Adding Relation: {relation}, No entity exists.")
            return

        trans_tuple = None
        check_subject_entity = self.entity_collection.get(ids = [subject_entity])
        if len(check_subject_entity["ids"]) == 0:
            self.logger.debug(f"Unknown Entity: {subject_entity} Found While Adding Relation: {relation}")
            trans_tuple = self.add_missing_entity(subject_entity)
            if trans_tuple:
                subject_entity = trans_tuple[1]
            
        trans_tuple = None
        check_object_entity = self.entity_collection.get(ids = [object_entity])
        if len(check_object_entity["ids"]) == 0:
            self.logger.debug(f"Unknown Entity: {object_entity} Found While Adding Relation: {relation}")
            trans_tuple = self.add_missing_entity(object_entity)
            if trans_tuple:
                object_entity = trans_tuple[1]
        
        existing_relations = self.relation_collection.get(
            where={
                "$or": [
                    {
                        "$and": [
                            {"subject_entity_id": {"$eq": subject_entity}},
                            {"object_entity_id": {"$eq": object_entity}}
                        ]
                    },
                    {
                        "$and": [
                            {"subject_entity_id": {"$eq": object_entity}},
                            {"object_entity_id": {"$eq": subject_entity}}
                        ]
                    }
                ]
            }
        )
        
        if existing_relations["ids"]:
            self.logger.debug(f"Same Relation: {existing_relations['ids']}.")
            return
        
        self.relation_collection.add(
            ids=[f"{subject_entity}_{object_entity}"],
            metadatas=[{
                "type": "relation",
                "subject_entity_id": subject_entity,
                "object_entity_id": object_entity
            }],
            documents=[relation_text],
            embeddings=[np.zeros(self.hidden_size, dtype=np.float32)]
        )


    def query_similar_entity(self, entity):
        entity_embedding = self.embedding.sentence_embedding(entity)
        
        results = self.entity_collection.query(
            query_embeddings=[entity_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )
        
        if results["ids"] and len(results["ids"][0]) > 0:
            similarity = 1 - results["distances"][0][0]
            
            if similarity >= self.synonym_threshold:
                entity_name = results["documents"][0][0]
                return entity_name
        
        return None



# new

    def get_path_id(self, node1_id:str, node2_id:str):
        relations = self.relation_collection.get(
            where={
                "$or": [
                    {
                        "$and": [
                            {"subject_entity_id": {"$eq": node1_id}},
                            {"object_entity_id": {"$eq": node2_id}}
                        ]
                    },
                    {
                        "$and": [
                            {"subject_entity_id": {"$eq": node2_id}},
                            {"object_entity_id": {"$eq": node1_id}}
                        ]
                    }
                ]
            }
        )

        if relations["ids"]:
            return ("relation", relations["ids"][0])
        else:
            connections = self.connection_collection.get(
                where={
                    "$or": [
                        {
                            "$and": [
                                {"entity_id": {"$eq": node1_id}},
                                {"chunk_id": {"$eq": node2_id}}
                            ]
                        },
                        {
                            "$and": [
                                {"entity_id": {"$eq": node2_id}},
                                {"chunk_id": {"$eq": node1_id}}
                            ]
                        }
                    ]
                }
            )

            if connections["ids"]:
                return ("connection", connections["ids"][0])
            else:
                raise Exception(f"Can not found path: {node1_id} to {node2_id}.")



    def enhance_edge_weight(self, query:str, path_list:List[Tuple[str,str]]):
        query_embedding = self.embedding.sentence_embedding(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        for path in path_list:
            self.logger.info(f"Enhance Path {path[1]}.")

            if path[0] == "relation":
                embedding = np.array(self.relation_collection.get(ids = path[1], include=['embeddings'])["embeddings"][0])
                new_embedding = embedding + query_embedding * edge_weight_coefficient(np.linalg.norm(embedding))
                self.relation_collection.update(
                    ids = [path[1]],
                    embeddings = [new_embedding]
                )
            else:
                embedding = np.array(self.connection_collection.get(ids = path[1], include=['embeddings'])["embeddings"][0])
                new_embedding = embedding + query_embedding * edge_weight_coefficient(np.linalg.norm(embedding))
                self.connection_collection.update(
                    ids = [path[1]],
                    embeddings = [new_embedding]
                )
    

    def punish_edge_weight(self, query:str, path_list:List[Tuple[str,str]]):
        # REMIND_PUNISH selects the punish write rule:
        #   off    — release code verbatim (kept only for explicit historical
        #                reproduction; gate/update are defective as audited).
        #   mirror — C semantics, the negative-memory completion:
        #                v <- v - beta * delta(||v||) * q_hat
        #                scalar gate s = v.q_hat >= 0 (stop once demoted below
        #                zero affinity — natural saturation). Pushes rejected
        #                edges BELOW cold start; punishes virgin edges.
        #   paper  — projection-vector update with a scalar direction gate:
        #                c = (v.q)/||q||^2; skip if c < 0
        #                v <- v - beta * (2/pi)cos((pi/2)c^2) * c * q.
        #                Since q is normalized above, c is the signed
        #                projection coefficient. The update uses c*q (a vector),
        #                never a coordinate-wise gate or scalar broadcast.
        # New callers get the paper-equation implementation. Historical runners
        # that explicitly set REMIND_PUNISH=off retain the original release path.
        mode = os.environ.get("REMIND_PUNISH", "paper")
        query_embedding = self.embedding.sentence_embedding(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        if mode in ("mirror", "paper"):
            beta = float(os.environ.get("REMIND_PUNISH_BETA", "1.0"))
            for path in path_list:
                if path[0] == "relation":
                    embedding = np.array(self.relation_collection.get(ids = path[1], include=['embeddings'])["embeddings"][0])
                else:
                    embedding = np.array(self.connection_collection.get(ids = path[1], include=['embeddings'])["embeddings"][0])
                if mode == "mirror":
                    s = float(np.dot(query_embedding, embedding))
                    if s < 0:
                        continue  # custom mirror rule saturates below zero affinity
                    coefficient = beta * abs(edge_weight_coefficient(np.linalg.norm(embedding)))
                    new_embedding = embedding - coefficient * query_embedding
                else:
                    # Gate on the scalar projection coefficient, not on each
                    # coordinate of c*q. Negative-alignment edges are already
                    # opposite the query direction and are left unchanged.
                    q_norm_sq = float(np.dot(query_embedding, query_embedding))
                    c = float(np.dot(embedding, query_embedding) / q_norm_sq)
                    if c < 0:
                        self.logger.info(
                            f"Skip Punish Path {path[1]} [paper] s={c:.3f} (<0)")
                        continue
                    delta = (2.0 / math.pi) * math.cos((math.pi / 2.0) * (c ** 2))
                    new_embedding = embedding - beta * delta * c * query_embedding
                    s = c
                    coefficient = beta * delta
                self.logger.info(f"Punish Path {path[1]} [{mode}] s={s:.3f} delta={coefficient:.3f}")
                if path[0] == "relation":
                    self.relation_collection.update(ids = [path[1]], embeddings = [new_embedding])
                else:
                    self.connection_collection.update(ids = [path[1]], embeddings = [new_embedding])
            return

        # ---- REMIND_PUNISH=off: release code verbatim for reproduction ----
        for path in path_list:
            self.logger.info(f"Punish Path {path[1]}.")

            if path[0] == "relation":
                embedding = np.array(self.relation_collection.get(ids = path[1], include=['embeddings'])["embeddings"][0])
                query_direction_vector = np.dot(query_embedding, embedding)

                if (np.dot(query_direction_vector, query_embedding) >= 0).all():
                    punish_coefficient = math.fabs(edge_weight_coefficient(np.linalg.norm(query_direction_vector)))
                    new_embedding = embedding - max(punish_coefficient,np.linalg.norm(query_direction_vector)) * query_direction_vector
                else:
                    continue
                self.relation_collection.update(
                    ids = [path[1]],
                    embeddings = [new_embedding]
                )
            else:
                embedding = np.array(self.connection_collection.get(ids = path[1], include=['embeddings'])["embeddings"][0])
                query_direction_vector = np.dot(query_embedding, embedding)
                if (np.dot(query_direction_vector, query_embedding) >= 0).all():
                    punish_coefficient = math.fabs(edge_weight_coefficient(np.linalg.norm(query_direction_vector)))
                    new_embedding = embedding - max(punish_coefficient,np.linalg.norm(query_direction_vector)) * query_direction_vector
                else:
                    continue
                self.connection_collection.update(
                    ids = [path[1]],
                    embeddings = [new_embedding]
                )

    def get_entity_edges(self, query:str, entity_id:str):
        """
    Retrieves edge information related to a specified entity and computes edge weights based on the query.
    
    Args:
        query: The query text string used to compute relevance weights.
        entity_id: The unique identifier of the target entity.
        
    Returns:
        Tuple[List[Dict], Dict]: A tuple containing two elements:
        
        1. processed_relations (List[Dict]): 
           A list of processed relation edges, where each element is a dictionary with the following keys:
           - "id": The unique ID of the relation edge.
           - "documents": The document content associated with the edge.
           - "from": The source entity ID (always equal to the input entity_id).
           - "to": The target entity ID.
           - "const_weight": The base similarity score between entities (computed via embedding dot product).
           - "weight": The composite weight score, calculated as:
             edge_weight_alpha * const_weight + (1 - edge_weight_alpha) * (query-to-edge-embedding similarity).
           
        2. processed_connection (Dict):
           A processed connection edge dictionary (may be empty), containing the following keys:
           - "id": The unique ID of the connection edge (if it exists).
           - "documents": The document content associated with the edge (if it exists).
           - "from": The entity ID (always equal to the input entity_id).
           - "to": The connected chunk ID.
           - "const_weight": The base similarity score between the entity and the chunk.
           - "weight": The composite weight score (computed in the same way as relation edges).
           
        Note: If no connection edge is found, processed_connection will be an empty dictionary.
    """
        query_embedding = self.embedding.sentence_embedding(query)

        relations = self.relation_collection.get(
            where={
                "$or": [
                    {"subject_entity_id": entity_id},
                    {"object_entity_id": entity_id}
                ]
            },
            include=['metadatas', 'documents', 'embeddings']
        )
        self.logger.trace(f"Found Relations: {len(relations['ids'])}")

        processed_relations = []
        for relation_iter in range(len(relations["ids"])):
            processed_relation_iter = {}
            processed_relation_iter["id"] = relations["ids"][relation_iter]
            processed_relation_iter["documents"] = relations["documents"][relation_iter]
            if relations["metadatas"][relation_iter]["subject_entity_id"] == entity_id:
                processed_relation_iter["from"] = relations["metadatas"][relation_iter]["subject_entity_id"]
                processed_relation_iter["to"] = relations["metadatas"][relation_iter]["object_entity_id"]
            else:
                processed_relation_iter["from"] = relations["metadatas"][relation_iter]["object_entity_id"]
                processed_relation_iter["to"] = relations["metadatas"][relation_iter]["subject_entity_id"]

            self.logger.trace(f"Processed Relation: {processed_relation_iter}")

            # A stale relation can survive extraction while one endpoint's
            # embedding is absent.  Treat it as an unusable edge instead of
            # indexing an empty Chroma result and aborting the whole question.
            subject_rows = self.entity_collection.get(
                ids=[processed_relation_iter["from"]], include=["embeddings"]
            ).get("embeddings")
            if subject_rows is None: subject_rows = []
            object_rows = self.entity_collection.get(
                ids=[processed_relation_iter["to"]], include=["embeddings"]
            ).get("embeddings")
            if object_rows is None: object_rows = []
            edge_rows = relations.get("embeddings")
            if edge_rows is None: edge_rows = []
            if len(subject_rows) == 0 or len(object_rows) == 0 or relation_iter >= len(edge_rows):
                self.logger.warning(
                    "Skipping relation with missing embedding: %s -> %s",
                    processed_relation_iter["from"], processed_relation_iter["to"],
                )
                continue
            subject_embedding = np.array(subject_rows[0])
            object_embedding = np.array(object_rows[0])
            edge_embedding = np.array(edge_rows[relation_iter])
            processed_relation_iter["const_weight"] = np.dot(subject_embedding, object_embedding)
            processed_relation_iter["weight"] = self.edge_weight_alpha * processed_relation_iter["const_weight"] + (1 - self.edge_weight_alpha) * np.dot(query_embedding,edge_embedding)
            
            processed_relations.append(processed_relation_iter)

        processed_connection = {}

        connection = self.connection_collection.get(where={"entity_id":entity_id}, include=['metadatas', 'documents', 'embeddings'])
        if connection["ids"]:
            processed_connection["id"] = connection["ids"][0]
            processed_connection["documents"] = connection["documents"][0]
            processed_connection["from"] = connection["metadatas"][0]["entity_id"]
            processed_connection["to"] = connection["metadatas"][0]["chunk_id"]
            entity_rows = self.entity_collection.get(
                ids=[processed_connection["from"]], include=["embeddings"]
            ).get("embeddings")
            if entity_rows is None: entity_rows = []
            chunk_rows = self.chunk_collection.get(
                ids=[processed_connection["to"]], include=["embeddings"]
            ).get("embeddings")
            if chunk_rows is None: chunk_rows = []
            connection_rows = connection.get("embeddings")
            if connection_rows is None: connection_rows = []
            if len(entity_rows) == 0 or len(chunk_rows) == 0 or len(connection_rows) == 0:
                self.logger.warning(
                    "Skipping connection with missing embedding: %s -> %s",
                    processed_connection["from"], processed_connection["to"],
                )
                return processed_relations, {}
            entity_embedding = np.array(entity_rows[0])
            chunk_embedding = np.array(chunk_rows[0])
            edge_embedding = np.array(connection_rows[0])
            processed_connection["const_weight"] = np.dot(entity_embedding, chunk_embedding)
            processed_connection["weight"] = self.edge_weight_alpha * processed_connection["const_weight"] + (1 - self.edge_weight_alpha) * np.dot(query_embedding,edge_embedding)
                


        return processed_relations, processed_connection
    

    def quick_query(self, query:str, entity_id):
        self.logger.debug(f"Start Quick Query: {query} in Entity: {entity_id}")

        self.dfs_entity = [entity_id]
        self.dfs_chunk = []
        self.dfs_edge = []

        self.strong_connection_dfs(query, entity_id)

        return self.dfs_entity, self.dfs_chunk, self.dfs_edge



        
    def strong_connection_dfs(self, query, entity_node_id):
        self.logger.trace(f"DFS Step in {entity_node_id}")
        relations, connection = self.get_entity_edges(query ,entity_node_id)
        if connection:
            if connection["weight"] > self.strong_connection_threshold:
                self.dfs_chunk.append(connection["to"])
                self.dfs_edge.append({"type":"connection","from":connection["from"],"to":connection["to"],"documents":connection["documents"],"id":connection["id"]})
        
        if relations:
            for relation in relations:
                if relation["to"] in self.dfs_entity:
                        continue
                if relation["weight"] > self.strong_connection_threshold:
                    self.dfs_entity.append(relation["to"])
                    self.dfs_edge.append({"type":"relation","from":relation["from"],"to":relation["to"],"documents":relation["documents"],"id":relation["id"]})
                    self.strong_connection_dfs(query, relation["to"])
                    
        



    


    

    # def get_chunk_document(self, chunk_id):
    #     chunk_data = self.chunk_collection.get(ids = [chunk_id], include=['documents'])
    #     return chunk_data["documents"][0]
    

    def get_all_entities(self):
        results = self.entity_collection.get()
        entities = {}
        
        for i, entity_id in enumerate(results["ids"]):
            entities[entity_id] = {
                "id": entity_id,
                "metadata": results["metadatas"][i] if "metadatas" in results else {},
                "text": results["documents"][i] if "documents" in results else ""
            }
        
        return entities
    
    def get_all_relations(self):
        results = self.relation_collection.get()
        relations = {}
        
        for i, relation_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if "metadatas" in results else {}
            relations[relation_id] = {
                "id": relation_id,
                "subject_entity_id": metadata.get("subject_entity_id", ""),
                "object_entity_id": metadata.get("object_entity_id", ""),
                "text": results["documents"][i] if "documents" in results else "",
                "metadata": metadata
            }
        
        return relations
