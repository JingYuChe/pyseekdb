"""
Collection get tests - testing collection.get() interface for all three modes
Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
import time
import json
import uuid
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pyseekdb
from typing import List, Union


# ==================== Simple 3D Embedding Function for Testing ====================
class Simple3DEmbeddingFunction:
    """Simple embedding function that returns 3-dimensional vectors for testing"""
    
    def __init__(self):
        self.dimension = 3
    
    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """Convert documents to 3D embeddings (simple hash-based)"""
        if isinstance(input, str):
            input = [input]
        
        embeddings = []
        for doc in input:
            # Simple hash-based 3D embedding for testing
            hash_val = hash(doc) % 1000
            embedding = [
                float((hash_val % 10) / 10.0),
                float(((hash_val // 10) % 10) / 10.0),
                float(((hash_val // 100) % 10) / 10.0)
            ]
            embeddings.append(embedding)
        
        return embeddings


# ==================== Environment Variable Configuration ====================
# Embedded mode
SEEKDB_PATH = os.environ.get('SEEKDB_PATH', os.path.join(project_root, "seekdb.db"))
SEEKDB_DATABASE = os.environ.get('SEEKDB_DATABASE', 'test')

# Server mode
SERVER_HOST = os.environ.get('SERVER_HOST', '127.0.0.1')
SERVER_PORT = int(os.environ.get('SERVER_PORT', '2881'))
SERVER_DATABASE = os.environ.get('SERVER_DATABASE', 'test')
SERVER_USER = os.environ.get('SERVER_USER', 'root')
SERVER_PASSWORD = os.environ.get('SERVER_PASSWORD', '')

# OceanBase mode
OB_HOST = os.environ.get('OB_HOST', 'localhost')
OB_PORT = int(os.environ.get('OB_PORT', '11202'))
OB_TENANT = os.environ.get('OB_TENANT', 'mysql')
OB_DATABASE = os.environ.get('OB_DATABASE', 'test')
OB_USER = os.environ.get('OB_USER', 'root')
OB_PASSWORD = os.environ.get('OB_PASSWORD', '')


class TestCollectionGet:
    """Test collection.get() interface for all three modes"""
    
    def _insert_test_data(self, client, collection_name: str):
        """Helper method to insert test data and return inserted IDs"""
        table_name = f"c$v1${collection_name}"
        
        # Insert test data with vectors, documents, and metadata
        test_data = [
            {
                "_id": str(uuid.uuid4()),
                "document": "This is a test document about machine learning",
                "embedding": [1.0, 2.0, 3.0],
                "metadata": {"category": "AI", "score": 95, "tag": "ml"}
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Python programming tutorial for beginners",
                "embedding": [2.0, 3.0, 4.0],
                "metadata": {"category": "Programming", "score": 88, "tag": "python"}
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Advanced machine learning algorithms",
                "embedding": [1.1, 2.1, 3.1],
                "metadata": {"category": "AI", "score": 92, "tag": "ml"}
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Data science with Python",
                "embedding": [2.1, 3.1, 4.1],
                "metadata": {"category": "Data Science", "score": 90, "tag": "python"}
            },
            {
                "_id": str(uuid.uuid4()),
                "document": "Introduction to neural networks",
                "embedding": [1.2, 2.2, 3.2],
                "metadata": {"category": "AI", "score": 85, "tag": "neural"}
            }
        ]
        
        # Store inserted IDs for return (using generated UUIDs)
        inserted_ids = []
        
        # Insert all data with generated UUIDs
        for data in test_data:
            # Use string ID directly (support any string format)
            id_str = data["_id"]
            inserted_ids.append(id_str)  # Store original ID string for return
            
            # Escape single quotes in ID
            id_str_escaped = id_str.replace("'", "''")
            
            # Convert vector to string format: [1.0,2.0,3.0]
            vector_str = "[" + ",".join(map(str, data["embedding"])) + "]"
            # Convert metadata to JSON string
            metadata_str = json.dumps(data["metadata"]).replace("'", "\\'")
            # Escape single quotes in document
            document_str = data["document"].replace("'", "\\'")
            
            # Use CAST to convert string to binary for varbinary(512) field
            sql = f"""INSERT INTO `{table_name}` (_id, document, embedding, metadata) 
                     VALUES (CAST('{id_str_escaped}' AS BINARY), '{document_str}', '{vector_str}', '{metadata_str}')"""
            client._server._execute(sql)
        
        return inserted_ids

    def _test_metadata_array_in_overlap(self, client):
        """Shared test flow for metadata array $in/$nin using JSON_OVERLAPS."""
        collection_name = f"test_tags_in_{int(time.time() * 1000)}"
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=pyseekdb.DefaultEmbeddingFunction(),
        )

        try:
            # Insert cases covering overlap, disjoint, missing, null, empty array
            collection.add(
                ids=["id_overlap", "id_disjoint", "id_missing", "id_null", "id_empty"],
                documents=["", "", "", "", ""],
                metadatas=[
                    {"category": "AI", "tags": ["ml", "ai"]},       # overlaps with ["ml", "python"]
                    {"category": "Web", "tags": ["java", "cpp"]},   # disjoint
                    {"category": "Missing"},                       # tags missing
                    {"category": "Null", "tags": None},            # tags explicit null
                    {"category": "Empty", "tags": []},             # tags empty array
                ],
            )

            # $in with overlap
            result = collection.get(
                where={"tags": {"$in": ["ml", "python"]}},
                include=["metadatas", "ids"],
            )
            assert result and "ids" in result
            assert set(result["ids"]) == {"id_overlap"}

            # $in with disjoint values -> no hits
            result = collection.get(
                where={"tags": {"$in": ["ruby"]}},
                include=["ids"],
            )
            assert result and "ids" in result
            assert len(result["ids"]) == 0

            # $in with empty list -> should return 0 rows
            result = collection.get(
                where={"tags": {"$in": []}},
                include=["ids"],
            )
            assert result and "ids" in result
            assert len(result["ids"]) == 0

            # $nin should keep disjoint/null/empty, exclude overlap; missing is excluded by JSON behavior
            result = collection.get(
                where={"tags": {"$nin": ["ml", "python"]}},
                include=["ids"],
            )
            assert result and "ids" in result
            assert set(result["ids"]) == {"id_disjoint", "id_null", "id_empty"}
        finally:
            try:
                client.delete_collection(name=collection_name)
            except Exception as cleanup_error:  # pragma: no cover
                print(f"Warning: cleanup failed for {collection_name}: {cleanup_error}")

    def _test_collection_get(self, collection, inserted_ids):
        # Test 1: Get by single ID
        print("\n✅ Testing get by single ID")
        results = collection.get(ids=inserted_ids[0])
        assert results is not None
        assert "ids" in results
        assert len(results["ids"]) == 1
        print(f"   Found {len(results['ids'])} result for ID={inserted_ids[0]}")

        # Test 2: Get by multiple IDs
        print("✅ Testing get by multiple IDs")
        if len(inserted_ids) >= 3:
            results = collection.get(ids=inserted_ids[:3])
            assert results is not None
            assert "ids" in results
            assert len(results["ids"]) <= 3
            print(f"   Found {len(results['ids'])} results for IDs={inserted_ids[:3]}")

        # Test 3: Get by metadata filter
        print("✅ Testing get with metadata filter (category=AI)")
        results = collection.get(where={"category": {"$eq": "AI"}}, limit=10)
        assert results is not None
        assert len(results["ids"]) > 0
        print(f"   Found {len(results['ids'])} results with category='AI'")

        # Test 4: Get with logical operators ($or)
        print("✅ Testing get with logical operators ($or)")
        results = collection.get(
            where={
                "$or": [
                    {"category": "AI"},
                    {"tag": "python"},
                ]
            },
            limit=10,
        )
        assert results is not None
        print(f"   Found {len(results['ids'])} results with $or condition")

        # Test 5: Get by document filter
        print("✅ Testing get with document filter")
        results = collection.get(where_document={"$contains": "machine learning"}, limit=10)
        assert results is not None
        print(f"   Found {len(results['ids'])} results containing 'machine learning'")

        # Test 6: Get with combined filters
        print("✅ Testing get with combined filters")
        results = collection.get(
            where={"category": {"$eq": "AI"}},
            where_document={"$contains": "machine"},
            limit=10,
        )
        assert results is not None
        print(f"   Found {len(results['ids'])} results matching all filters")

        # Test 7: Get with limit and offset
        print("✅ Testing get with limit and offset")
        results = collection.get(limit=3, offset=0)
        assert results is not None
        assert "ids" in results
        assert len(results["ids"]) <= 3
        print(f"   Found {len(results['ids'])} results (limit=3, offset=0)")

        # Test 8: Get all data without filters
        print("✅ Testing get all data without filters")
        results = collection.get(limit=100)
        assert results is not None
        assert len(results["ids"]) > 0
        print(f"   Found {len(results['ids'])} total results")

        # Test 9: Get with include parameter
        print("✅ Testing get with include parameter")
        results = collection.get(
            ids=inserted_ids[:2],
            include=["documents", "metadatas"],
        )
        assert results is not None
        assert isinstance(results, dict), "Should return dict"
        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert len(results["ids"]) == 2
        print(f"   Found {len(results['ids'])} results with documents and metadatas")

        # Test 10: Get by multiple IDs (should return dict)
        print("✅ Testing get by multiple IDs (returns dict)")
        if len(inserted_ids) >= 3:
            results = collection.get(ids=inserted_ids[:3])
            assert results is not None
            assert isinstance(results, dict), "Should return dict"
            assert "ids" in results
            assert len(results["ids"]) <= 3
            print(f"   Found {len(results['ids'])} results for {len(inserted_ids[:3])} IDs")

        # Test 11: Single ID returns dict format
        print("✅ Testing single ID returns dict format")
        results = collection.get(ids=inserted_ids[0])
        assert results is not None
        assert isinstance(results, dict), "Should return dict"
        assert "ids" in results
        assert len(results["ids"]) == 1
        print(f"   Single result with {len(results['ids'])} item")

        # Test 12: Get with filters returns dict format
        print("✅ Testing get with filters returns dict format")
        results = collection.get(where={"category": {"$eq": "AI"}}, limit=10)
        assert results is not None
        assert isinstance(results, dict), "Should return dict"
        assert "ids" in results
        print(f"   Found {len(results['ids'])} items matching filter")

        # Test 13: Get with scalar $in operator
        print("✅ Testing get with scalar $in operator")
        results = collection.get(where={"tag": {"$in": ["ml", "python"]}}, limit=10)
        assert results is not None
        assert "ids" in results
        assert len(results["ids"]) > 0
        print(f"   Found {len(results['ids'])} results with tag in ['ml', 'python']")

        # Test 14: Get with scalar $nin operator
        print("✅ Testing get with scalar $nin operator")
        results = collection.get(where={"tag": {"$nin": ["ml", "python"]}}, limit=10)
        assert results is not None
        assert "ids" in results
        # Should return rows with tag='neural' (excluded 'ml' and 'python')
        print(f"   Found {len(results['ids'])} results with tag not in ['ml', 'python']")

    def test_embedded_metadata_array_in_overlap(self):
        """
        Regression test for JSON array $in operator with embedded client.
        Includes overlap, disjoint, empty list, and complementary $nin expectations.
        """
        try:
            import pylibseekdb  # noqa: F401
        except ImportError:
            pytest.skip("seekdb embedded package is not installed")

        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )
        self._test_metadata_array_in_overlap(client)
    
    def test_server_metadata_array_in_overlap(self):
        """
        Regression test for JSON array $in operator with server client.
        Includes overlap, disjoint, empty list, and complementary $nin expectations.
        """
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )

        # Test connection
        try:
            result = client._server._execute("SELECT 1 as test")
            assert result and result[0].get("test") == 1
        except Exception as exc:
            pytest.fail(f"seekdb server connection failed ({SERVER_HOST}:{SERVER_PORT}): {exc}")
        self._test_metadata_array_in_overlap(client)
    
    def test_oceanbase_metadata_array_in_overlap(self):
        """
        Regression test for JSON array $in operator with OceanBase client.
        Includes overlap, disjoint, empty list, and complementary $nin expectations.
        """
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )

        # Test connection
        try:
            result = client._server._execute("SELECT 1 as test")
            assert result and result[0].get("test") == 1
        except Exception as exc:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {exc}")
        self._test_metadata_array_in_overlap(client)
    
    def test_embedded_metadata_array_nin_overlap(self):
        """
        Regression test for JSON array $nin operator with embedded client.
        Tests that tags=["ml","ai"] is excluded by $nin ["ml","python"] via JSON_OVERLAPS.
        """
        try:
            import pylibseekdb  # noqa: F401
        except ImportError:
            pytest.skip("seekdb embedded package is not installed")

        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )

        collection_name = f"test_tags_nin_{int(time.time() * 1000)}"
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=pyseekdb.DefaultEmbeddingFunction(),
        )

        try:
            # Insert two records: one with overlap, one without
            collection.add(
                ids=["id1", "id2"],
                documents=["", ""],
                metadatas=[
                    {"category": "AI", "tags": ["ml", "ai"]},  # has overlap with ["ml", "python"]
                    {"category": "Web", "tags": ["java", "cpp"]},  # no overlap with ["ml", "python"]
                ],
            )

            result = collection.get(
                where={"tags": {"$nin": ["ml", "python"]}},
                include=["metadatas", "ids"],
            )

            # Expect 1 row (id2) because its tags have no overlap with the exclusion list
            assert result is not None
            assert "ids" in result
            assert len(result["ids"]) == 1
            assert "id2" in result["ids"]
            assert "id1" not in result["ids"]
            assert result["metadatas"][0].get("tags") == ["java", "cpp"]
        finally:
            try:
                client.delete_collection(name=collection_name)
            except Exception as cleanup_error:  # pragma: no cover
                print(f"Warning: cleanup failed for {collection_name}: {cleanup_error}")
    
    def test_server_metadata_array_nin_overlap(self):
        """
        Regression test for JSON array $nin operator with server client.
        Tests that tags=["ml","ai"] is excluded by $nin ["ml","python"] via JSON_OVERLAPS.
        """
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )

        # Test connection
        try:
            result = client._server._execute("SELECT 1 as test")
            assert result and result[0].get("test") == 1
        except Exception as exc:
            pytest.fail(f"seekdb server connection failed ({SERVER_HOST}:{SERVER_PORT}): {exc}")

        collection_name = f"test_tags_nin_{int(time.time() * 1000)}"
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=pyseekdb.DefaultEmbeddingFunction(),
        )

        try:
            # Insert two records: one with overlap, one without
            collection.add(
                ids=["id1", "id2"],
                documents=["", ""],
                metadatas=[
                    {"category": "AI", "tags": ["ml", "ai"]},  # has overlap with ["ml", "python"]
                    {"category": "Web", "tags": ["java", "cpp"]},  # no overlap with ["ml", "python"]
                ],
            )

            result = collection.get(
                where={"tags": {"$nin": ["ml", "python"]}},
                include=["metadatas", "ids"],
            )

            # Expect 1 row (id2) because its tags have no overlap with the exclusion list
            assert result is not None
            assert "ids" in result
            assert len(result["ids"]) == 1
            assert "id2" in result["ids"]
            assert "id1" not in result["ids"]
            assert result["metadatas"][0].get("tags") == ["java", "cpp"]
        finally:
            try:
                client.delete_collection(name=collection_name)
            except Exception as cleanup_error:  # pragma: no cover
                print(f"Warning: cleanup failed for {collection_name}: {cleanup_error}")
    
    def test_oceanbase_metadata_array_nin_overlap(self):
        """
        Regression test for JSON array $nin operator with OceanBase client.
        Tests that tags=["ml","ai"] is excluded by $nin ["ml","python"] via JSON_OVERLAPS.
        """
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )

        # Test connection
        try:
            result = client._server._execute("SELECT 1 as test")
            assert result and result[0].get("test") == 1
        except Exception as exc:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {exc}")

        collection_name = f"test_tags_nin_{int(time.time() * 1000)}"
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=pyseekdb.DefaultEmbeddingFunction(),
        )

        try:
            # Insert two records: one with overlap, one without
            collection.add(
                ids=["id1", "id2"],
                documents=["", ""],
                metadatas=[
                    {"category": "AI", "tags": ["ml", "ai"]},  # has overlap with ["ml", "python"]
                    {"category": "Web", "tags": ["java", "cpp"]},  # no overlap with ["ml", "python"]
                ],
            )

            result = collection.get(
                where={"tags": {"$nin": ["ml", "python"]}},
                include=["metadatas", "ids"],
            )

            # Expect 1 row (id2) because its tags have no overlap with the exclusion list
            assert result is not None
            assert "ids" in result
            assert len(result["ids"]) == 1
            assert "id2" in result["ids"]
            assert "id1" not in result["ids"]
            assert result["metadatas"][0].get("tags") == ["java", "cpp"]
        finally:
            try:
                client.delete_collection(name=collection_name)
            except Exception as cleanup_error:  # pragma: no cover
                print(f"Warning: cleanup failed for {collection_name}: {cleanup_error}")
    
    def test_embedded_collection_get(self):
        """Test collection.get() with embedded client"""
        # Check if seekdb package is available
        try:
            import pylibseekdb
        except ImportError:
            pytest.fail("seekdb embedded package is not installed")
        
        # Create embedded client
        client = pyseekdb.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.SeekdbEmbeddedClient)
        
        # Create test collection
        collection_name = f"test_get_{int(time.time())}"
        from pyseekdb import HNSWConfiguration
        config = HNSWConfiguration(dimension=3, distance='l2')
        # Use a simple 3D embedding function to match the dimension
        embedding_function = Simple3DEmbeddingFunction()
        collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=embedding_function
        )
        
        try:
            inserted_ids = self._insert_test_data(client, collection_name)
            assert len(inserted_ids) > 0, f"Failed to get inserted IDs. Expected at least 1, got {len(inserted_ids)}"
            if len(inserted_ids) < 5:
                print(f"   Warning: Expected 5 inserted IDs, but got {len(inserted_ids)}")

            self._test_collection_get(collection, inserted_ids)
            
        finally:
            # Cleanup
            try:
                client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")
    
    def test_server_collection_get(self):
        """Test collection.get() with server client"""
        # Create server client
        client = pyseekdb.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            tenant="sys",  # Default tenant for seekdb Server
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.RemoteServerClient)
        
        # Test connection
        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"Server connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_get_{int(time.time())}"
        from pyseekdb import HNSWConfiguration
        config = HNSWConfiguration(dimension=3, distance='l2')
        # Use a simple 3D embedding function to match the dimension
        embedding_function = Simple3DEmbeddingFunction()
        collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=embedding_function
        )
        
        try:
            # Insert test data and get IDs
            inserted_ids = self._insert_test_data(client, collection_name)
            assert len(inserted_ids) > 0, f"Failed to get inserted IDs. Expected at least 1, got {len(inserted_ids)}"
            if len(inserted_ids) < 5:
                print(f"   Warning: Expected 5 inserted IDs, but got {len(inserted_ids)}")

            self._test_collection_get(collection, inserted_ids)
            
        finally:
            # Cleanup
            try:
                client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")
    
    def test_oceanbase_collection_get(self):
        """Test collection.get() with OceanBase client"""
        # Create OceanBase client
        client = pyseekdb.Client(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, pyseekdb.RemoteServerClient)
        
        # Test connection
        try:
            result = client._server._execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.fail(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_get_{int(time.time())}"
        from pyseekdb import HNSWConfiguration
        config = HNSWConfiguration(dimension=3, distance='l2')
        # Use a simple 3D embedding function to match the dimension
        embedding_function = Simple3DEmbeddingFunction()
        collection = client.create_collection(
            name=collection_name,
            configuration=config,
            embedding_function=embedding_function
        )
        
        try:
            # Insert test data and get IDs
            inserted_ids = self._insert_test_data(client, collection_name)
            assert len(inserted_ids) > 0, f"Failed to get inserted IDs. Expected at least 1, got {len(inserted_ids)}"
            if len(inserted_ids) < 5:
                print(f"   Warning: Expected 5 inserted IDs, but got {len(inserted_ids)}")
            
            self._test_collection_get(collection, inserted_ids)
            
        finally:
            # Cleanup
            try:
                client.delete_collection(name=collection_name)
                print(f"   Cleaned up collection: {collection_name}")
            except Exception as cleanup_error:
                print(f"   Warning: Failed to cleanup collection: {cleanup_error}")
            pass


if __name__ == "__main__":
    print("\n" + "="*60)
    print("pyseekdb - Collection Get Tests")
    print("="*60)
    print("\nEnvironment Variable Configuration:")
    print(f"  Embedded mode: path={SEEKDB_PATH}, database={SEEKDB_DATABASE}")
    print(f"  Server mode: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}/{SERVER_DATABASE}")
    print(f"  OceanBase mode: {OB_USER}@{OB_TENANT} -> {OB_HOST}:{OB_PORT}/{OB_DATABASE}")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "-s"])

