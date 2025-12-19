"""
Empty value handling tests - testing upsert operations with empty strings, empty lists, and None values
Supports configuring connection parameters via environment variables
Tests the fixes for falsy value handling bugs in _collection_upsert method
"""
import pytest
import sys
import os
import time
import uuid
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pyseekdb
from pyseekdb.client.meta_info import CollectionNames, CollectionFieldNames


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


class TestEmptyValueHandling:
    """Test empty value handling in upsert operations for all three modes"""
    
    def _cleanup_collection(self, client, name: str):
        """Clean up test collection"""
        try:
            client.delete_collection(name=name)
        except Exception as cleanup_error:  # pragma: no cover - best effort cleanup
            print(f"Warning: failed to cleanup collection '{name}': {cleanup_error}")
    
    def _create_collection(self, client, mode_name: str):
        """Create test collection with unique name"""
        collection_name = f"test_empty_values_{mode_name}_{int(time.time() * 1000)}"
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=pyseekdb.DefaultEmbeddingFunction()
        )
        return collection_name, collection
    
    def test_embedded_empty_value_handling(self):
        """Test empty value handling with embedded client (SeekdbEmbedded)"""
        try:
            import pylibseekdb  # noqa: F401
        except ImportError:
            pytest.fail("seekdb embedded package is not installed")
        
        client = pyseekdb.Client(path=SEEKDB_PATH, database=SEEKDB_DATABASE)
        collection_name, collection = self._create_collection(client, "embedded")
        
        try:
            self._run_empty_value_tests(collection, "embedded")
        finally:
            self._cleanup_collection(client, collection_name)
    
    def test_server_empty_value_handling(self):
        """Test empty value handling with seekdb server (RemoteServerClient default tenant)"""
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
        
        collection_name, collection = self._create_collection(client, "server")
        
        try:
            self._run_empty_value_tests(collection, "server")
        finally:
            self._cleanup_collection(client, collection_name)
    
    def test_oceanbase_empty_value_handling(self):
        """Test empty value handling with OceanBase deployment"""
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
        
        collection_name, collection = self._create_collection(client, "oceanbase")
        
        try:
            self._run_empty_value_tests(collection, "oceanbase")
        finally:
            self._cleanup_collection(client, collection_name)
    
    def _run_empty_value_tests(self, collection, mode_name):
        """Run empty value handling tests for a given collection"""
        print(f"\n🧪 Testing empty value handling for {mode_name} client")
        
        # Test 1: Upsert update path - empty string document (Line 1072 fix)
        self._test_upsert_update_empty_document(collection, mode_name)
        
        # Test 2: Upsert insert path - empty string document (Line 1093 fix)  
        self._test_upsert_insert_empty_document(collection, mode_name)
        
        # Test 3: Upsert update path - empty metadata (should work correctly)
        self._test_upsert_empty_metadata(collection, mode_name)
        
        # Test 4: Add method baseline - should work correctly
        self._test_add_empty_values_baseline(collection, mode_name)
        
        print(f"✅ All empty value tests passed for {mode_name} client")
    
    def _test_upsert_update_empty_document(self, collection, mode_name):
        """Test upsert update path with empty string document (Line 1072 fix)"""
        print(f"\n🔍 Testing upsert update path - empty document ({mode_name})")
        
        # Add initial document
        test_id = f"update_test_{int(time.time() * 1000)}"
        collection.add(
            ids=[test_id],
            documents=["original document"],
            metadatas=[{"test": "update_path"}]
        )
        
        # Test cases for empty document values
        test_cases = [
            ("", "empty string"),
            ("   ", "whitespace string"),
            ("0", "string zero"),
            ("false", "string false"),
        ]
        
        for doc_value, description in test_cases:
            print(f"   Testing {description}: {repr(doc_value)}")
            
            # Upsert with empty/falsy document value (triggers Line 1072 fix)
            collection.upsert(
                ids=[test_id],
                documents=[doc_value],
                metadatas=[{"test": "updated", "case": description}]
            )
            
            # Verify the document was stored correctly
            results = collection.get(ids=[test_id], include=["documents", "metadatas"])
            assert len(results["ids"]) == 1, f"Should find exactly one result for {description}"
            
            actual_doc = results["documents"][0]
            assert actual_doc == doc_value, f"Expected {repr(doc_value)}, got {repr(actual_doc)} for {description}"
            
            # Ensure it's not the literal string 'NULL'
            assert actual_doc != "NULL", f"Document should not be literal 'NULL' string for {description}"
            
            print(f"   ✅ {description} correctly stored as {repr(actual_doc)}")
    
    def _test_upsert_insert_empty_document(self, collection, mode_name):
        """Test upsert insert path with empty string document (Line 1093 fix)"""
        print(f"\n🔍 Testing upsert insert path - empty document ({mode_name})")
        
        # Test cases for empty document values in insert path
        test_cases = [
            ("", "empty string"),
            ("   ", "whitespace string"),
            ("0", "string zero"),
            ("false", "string false"),
        ]
        
        for i, (doc_value, description) in enumerate(test_cases):
            test_id = f"insert_test_{mode_name}_{i}_{int(time.time() * 1000)}"
            print(f"   Testing {description}: {repr(doc_value)}")
            
            # Upsert non-existing record (triggers Line 1093 fix)
            collection.upsert(
                ids=[test_id],
                documents=[doc_value],
                metadatas=[{"test": "insert_path", "case": description}]
            )
            
            # Verify the document was stored correctly
            results = collection.get(ids=[test_id], include=["documents", "metadatas"])
            assert len(results["ids"]) == 1, f"Should create exactly one result for {description}"
            
            actual_doc = results["documents"][0]
            assert actual_doc == doc_value, f"Expected {repr(doc_value)}, got {repr(actual_doc)} for {description}"
            
            # Ensure it's not None/NULL
            assert actual_doc is not None, f"Document should not be None for {description}"
            
            print(f"   ✅ {description} correctly stored as {repr(actual_doc)}")
    
    def _test_upsert_empty_metadata(self, collection, mode_name):
        """Test upsert with empty metadata (should work correctly)"""
        print(f"\n🔍 Testing upsert empty metadata ({mode_name})")
        
        test_id = f"meta_test_{int(time.time() * 1000)}"
        
        # Add initial document
        collection.add(
            ids=[test_id],
            documents=["test document"],
            metadatas=[{"initial": "value"}]
        )
        
        # Test cases for metadata values
        test_cases = [
            ({}, "empty dict"),
            ({"": ""}, "dict with empty string key/value"),
            ({"key": ""}, "dict with empty string value"),
            ({"key": None}, "dict with None value"),
            ({"key": 0}, "dict with zero value"),
            ({"key": False}, "dict with False value"),
        ]
        
        for meta_value, description in test_cases:
            print(f"   Testing {description}: {meta_value}")
            
            # Upsert with test metadata
            collection.upsert(
                ids=[test_id],
                documents=["updated document"],
                metadatas=[meta_value]
            )
            
            # Verify the metadata was stored correctly
            results = collection.get(ids=[test_id], include=["documents", "metadatas"])
            assert len(results["ids"]) == 1, f"Should find exactly one result for {description}"
            
            actual_meta = results["metadatas"][0]
            assert actual_meta == meta_value, f"Expected {meta_value}, got {actual_meta} for {description}"
            
            print(f"   ✅ {description} correctly stored as {actual_meta}")
    
    def _test_add_empty_values_baseline(self, collection, mode_name):
        """Test add method with empty values as baseline (should work correctly)"""
        print(f"\n🔍 Testing add method baseline - empty values ({mode_name})")
        
        # Test cases for add method with empty values
        test_cases = [
            ("", "empty string"),
            ("   ", "whitespace string"),
            ("0", "string zero"),
        ]
        
        for i, (doc_value, description) in enumerate(test_cases):
            test_id = f"add_baseline_{mode_name}_{i}_{int(time.time() * 1000)}"
            print(f"   Testing {description}: {repr(doc_value)}")
            
            # Add with empty document value (baseline test)
            collection.add(
                ids=[test_id],
                documents=[doc_value],
                metadatas=[{"test": "add_baseline", "case": description}]
            )
            
            # Verify the document was stored correctly
            results = collection.get(ids=[test_id], include=["documents", "metadatas"])
            assert len(results["ids"]) == 1, f"Should find exactly one result for {description}"
            
            actual_doc = results["documents"][0]
            assert actual_doc == doc_value, f"Expected {repr(doc_value)}, got {repr(actual_doc)} for {description}"
            
            print(f"   ✅ {description} correctly stored as {repr(actual_doc)}")
    
    def test_edge_cases_and_regression(self):
        """Test edge cases and regression scenarios using embedded client"""
        try:
            import pylibseekdb  # noqa: F401
        except ImportError:
            pytest.fail("seekdb embedded package is not installed")
        
        client = pyseekdb.Client(path=SEEKDB_PATH, database=SEEKDB_DATABASE)
        collection_name, collection = self._create_collection(client, "edge_cases")
        
        try:
            self._test_mixed_empty_values_batch(collection)
        finally:
            self._cleanup_collection(client, collection_name)
    
    def _test_mixed_empty_values_batch(self, collection):
        """Test mixed empty and non-empty values in batch operations"""
        print("🔍 Testing mixed empty and non-empty values in batch")
        
        test_ids = [f"mixed_{i}_{int(time.time() * 1000)}" for i in range(4)]
        test_docs = ["normal doc", "", "   ", "another normal doc"]
        test_metas = [{"type": "normal"}, {}, {"empty": ""}, {"type": "normal2"}]
        
        # Add initial batch
        collection.add(
            ids=test_ids,
            documents=test_docs,
            metadatas=test_metas
        )
        
        # Upsert with mixed empty values
        updated_docs = ["", "updated normal", "", "final doc"]
        updated_metas = [{"updated": True}, {}, {"key": ""}, {"final": True}]
        
        collection.upsert(
            ids=test_ids,
            documents=updated_docs,
            metadatas=updated_metas
        )
        
        # Verify all values were stored correctly
        results = collection.get(ids=test_ids, include=["documents", "metadatas"])
        assert len(results["ids"]) == 4, "Should find all 4 results"
        
        for i, (expected_doc, expected_meta) in enumerate(zip(updated_docs, updated_metas)):
            actual_doc = results["documents"][i]
            actual_meta = results["metadatas"][i]
            
            assert actual_doc == expected_doc, f"Document {i}: expected {repr(expected_doc)}, got {repr(actual_doc)}"
            assert actual_meta == expected_meta, f"Metadata {i}: expected {expected_meta}, got {actual_meta}"
        
        print("✅ Mixed empty and non-empty values handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
