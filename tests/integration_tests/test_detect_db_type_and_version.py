"""
Tests for detect_db_type_and_version method
Tests database type and version detection functionality for server and oceanbase client modes
Refactored to use client fixtures for parameterized testing
"""
import pytest

import pyseekdb
from pyseekdb.client.version import Version


class TestDetectDbTypeAndVersion:
    """Tests for detect_db_type_and_version method"""
    
    def test_seekdb_type_detection(self, server_client):
        """Test: detect seekdb Server type and version"""
        # Verify client type
        assert server_client is not None
        assert hasattr(server_client, '_server')
        assert isinstance(server_client._server, pyseekdb.RemoteServerClient)
        
        # Test detect_db_type_and_version
        db_type, version = server_client._server.detect_db_type_and_version()
        
        # Verify results
        assert db_type == "seekdb"
        assert version is not None
        assert isinstance(version, Version)
        # Test version comparison
        assert version > Version("0.0.0.0"), f"Version should be greater than 0.0.0.0, got: {version}"
        
        print(f"\n✅ Successfully detected seekdb Server")
        print(f"   Database type: {db_type}")
        print(f"   Version: {version}")
    
    def test_ob_type_detection(self, oceanbase_client):
        """Test: detect OceanBase Server type and version"""
        # Verify client type
        assert oceanbase_client is not None
        assert hasattr(oceanbase_client, '_server')
        assert isinstance(oceanbase_client._server, pyseekdb.RemoteServerClient)
        
        # Test detect_db_type_and_version
        db_type, version = oceanbase_client._server.detect_db_type_and_version()
        
        # Verify results
        assert db_type == "oceanbase"
        assert version is not None
        assert isinstance(version, Version)
        # Test version comparison
        assert version > Version("0.0.0.0"), f"Version should be greater than 0.0.0.0, got: {version}"
        
        print(f"\n✅ Successfully detected OceanBase Server")
        print(f"   Database type: {db_type}")
        print(f"   Version: {version}")
    
    def test_connection_establishment(self, server_client):
        """Test: verify detect_db_type_and_version establishes connection automatically"""
        # Note: server_client from fixture may already be connected due to connection test
        # We test that the method works correctly
        
        # Call detect_db_type_and_version
        db_type, version = server_client._server.detect_db_type_and_version()
        
        # Verify connection is established
        assert server_client._server.is_connected()
        
        # Verify results
        assert db_type in ["seekdb", "oceanbase"]
        assert version is not None
        
        print(f"\n✅ detect_db_type_and_version successfully works with connection")
        print(f"   Database type: {db_type}")
        print(f"   Version: {version}")
    
    def test_return_format(self, server_client):
        """Test: verify detect_db_type_and_version returns correct tuple format"""
        # Test detect_db_type_and_version
        result = server_client._server.detect_db_type_and_version()
        
        # Verify return type is tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        db_type, version = result
        
        # Verify tuple elements
        assert isinstance(db_type, str)
        assert isinstance(version, Version)
        assert db_type in ["seekdb", "oceanbase"]
        assert version > Version("0.0.0.0")
        
        print(f"\n✅ detect_db_type_and_version returns correct tuple format")
        print(f"   Result: {result}")
        print(f"   Type: {type(result)}")
        print(f"   Length: {len(result)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
