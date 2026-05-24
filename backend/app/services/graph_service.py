import os
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import select

logger = logging.getLogger("sentinel.graph_service")

try:
    # Try importing Neo4j and Milvus drivers if available
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except Exception as e:
    HAS_NEO4J = False

try:
    from pymilvus import connections, utility, Collection
    HAS_MILVUS = True
except Exception as e:
    HAS_MILVUS = False

class GraphService:
    def __init__(self):
        self.neo4j_driver = None
        if HAS_NEO4J:
            try:
                uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                user = os.getenv("NEO4J_USER", "neo4j")
                pwd = os.getenv("NEO4J_PASSWORD", "password")
                self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, pwd))
                logger.info("Neo4j database connection established successfully.")
            except Exception as e:
                logger.warning(f"Could not connect to Neo4j: {str(e)}. Using fallback database registry.")
                self.neo4j_driver = None

        if HAS_MILVUS:
            try:
                connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port=os.getenv("MILVUS_PORT", "19530"))
                logger.info("Milvus vector database connection established.")
            except Exception as e:
                logger.warning(f"Could not connect to Milvus: {str(e)}.")

    async def register_identity(self, case_id: str, name: str, dob: str, face_hashes: List[str], voice_hashes: List[str]) -> None:
        """
        Creates an identity node in Neo4j and updates Milvus face/voice vector catalogs.
        """
        logger.info(f"Registering identity nodes for Case '{case_id}' (Name: {name})...")
        
        if self.neo4j_driver:
            try:
                # Synchronous session write, running in background executor would be better but simple write here:
                with self.neo4j_driver.session() as session:
                    session.run(
                        "MERGE (i:Identity {case_id: $case_id}) "
                        "SET i.name = $name, i.dob = $dob "
                        "WITH i "
                        "UNWIND $face_hashes as fh "
                        "MERGE (f:FaceEmbedding {hash: fh}) "
                        "MERGE (i)-[:HAS_FACE]->(f) "
                        "WITH i "
                        "UNWIND $voice_hashes as vh "
                        "MERGE (v:VoiceEmbedding {hash: vh}) "
                        "MERGE (i)-[:HAS_VOICE]->(v)",
                        case_id=case_id, name=name, dob=dob,
                        face_hashes=face_hashes, voice_hashes=voice_hashes
                    )
                logger.info(f"Graph nodes merged successfully in Neo4j.")
            except Exception as e:
                logger.error(f"Failed to execute Neo4j write transaction: {str(e)}")

    async def find_linked_fraud_cases(
        self, case_id: str, current_name: str, face_hashes: List[str], voice_hashes: List[str], db_session: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries Neo4j (or falls back to relational SQLite checks) to look for cases
        reusing face/voice embedding fingerprints under different demographic names.
        """
        logger.info("Auditing network for linked fraud circles...")
        linked_cases = []

        # 1. Primary path: Neo4j Cypher query
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    # Query to find different cases sharing same face embedding
                    result = session.run(
                        "MATCH (i:Identity)-[:HAS_FACE]->(f:FaceEmbedding) "
                        "WHERE f.hash IN $face_hashes AND i.case_id <> $case_id AND i.name <> $current_name "
                        "RETURN i.case_id AS case_id, i.name AS name, f.hash AS hash, 'FACE' AS match_type "
                        "UNION "
                        "MATCH (i:Identity)-[:HAS_VOICE]->(v:VoiceEmbedding) "
                        "WHERE v.hash IN $voice_hashes AND i.case_id <> $case_id AND i.name <> $current_name "
                        "RETURN i.case_id AS case_id, i.name AS name, v.hash AS hash, 'VOICE' AS match_type",
                        face_hashes=face_hashes, voice_hashes=voice_hashes,
                        case_id=case_id, current_name=current_name
                    )
                    for record in result:
                        linked_cases.append({
                            "case_id": record["case_id"],
                            "name": record["name"],
                            "matched_hash": record["hash"],
                            "match_type": record["match_type"]
                        })
                if linked_cases:
                    logger.info(f"Neo4j link analysis discovered {len(linked_cases)} fraud intersections!")
                    return linked_cases
            except Exception as e:
                logger.error(f"Neo4j link query failed: {str(e)}")

        # 2. Relational fallback: Scan SQLite historical records
        if db_session:
            try:
                from backend.app.models.db_models import Case
                # Select completed cases excluding current case
                stmt = select(Case).where(Case.status == "COMPLETED")
                res = await db_session.execute(stmt)
                cases = res.scalars().all()
                
                for c in cases:
                    if str(c.id) == case_id:
                        continue
                        
                    payload = c.ocr_payload or {}
                    hist_name = payload.get("full_name")
                    if not hist_name or hist_name == current_name:
                        continue
                        
                    hist_faces = payload.get("face_embeddings_hashes", [])
                    hist_voices = payload.get("voice_embeddings_hashes", [])
                    
                    # Verify face embedding overlap
                    for fh in face_hashes:
                        if fh in hist_faces:
                            linked_cases.append({
                                "case_id": str(c.id),
                                "name": hist_name,
                                "matched_hash": fh,
                                "match_type": "FACE"
                            })
                            break
                            
                    # Verify voice embedding overlap
                    for vh in voice_hashes:
                        if vh in hist_voices:
                            linked_cases.append({
                                "case_id": str(c.id),
                                "name": hist_name,
                                "matched_hash": vh,
                                "match_type": "VOICE"
                            })
                            break
            except Exception as e:
                logger.warning(f"Relational fallback for graph service failed: {str(e)}")
                
        return linked_cases

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()
