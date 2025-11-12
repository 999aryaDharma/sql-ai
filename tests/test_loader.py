"""Tests for schema loader."""
import pytest
from pathlib import Path
from sqlgenx.core.schema_loader import SchemaLoader, SchemaInfo, TableInfo


def test_parse_simple_schema():
    """Test parsing a simple CREATE TABLE statement."""
    sql = """
    CREATE TABLE users (
        id INT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(255) UNIQUE
    );
    """
    
    loader = SchemaLoader()
    schema_info = loader.parse_schema(sql)
    
    assert isinstance(schema_info, SchemaInfo)
    assert len(schema_info.tables) == 1
    
    table = schema_info.tables[0]
    assert table.name == "users"
    assert len(table.columns) == 3


def test_parse_with_foreign_keys():
    """Test parsing tables with foreign key relationships."""
    sql = """
    CREATE TABLE departments (
        dept_id INT PRIMARY KEY,
        dept_name VARCHAR(100)
    );
    
    CREATE TABLE employees (
        emp_id INT PRIMARY KEY,
        emp_name VARCHAR(100),
        dept_id INT,
        FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
    );
    """
    
    loader = SchemaLoader()
    schema_info = loader.parse_schema(sql)
    
    assert len(schema_info.tables) == 2


def test_table_to_text():
    """Test converting table info to text representation."""
    sql = """
    CREATE TABLE products (
        product_id INT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        price DECIMAL(10, 2)
    );
    """
    
    loader = SchemaLoader()
    schema_info = loader.parse_schema(sql)
    
    if schema_info.tables:
        text = schema_info.tables[0].to_text()
        assert "Table: products" in text
        assert "product_id" in text
        assert "PRIMARY KEY" in text


def test_schema_to_text_chunks():
    """Test converting schema to text chunks for embedding."""
    sql = """
    CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100));
    CREATE TABLE posts (id INT PRIMARY KEY, user_id INT);
    """
    
    loader = SchemaLoader()
    schema_info = loader.parse_schema(sql)
    chunks = schema_info.to_text_chunks()
    
    assert len(chunks) == 2
    assert all("table" in chunk for chunk in chunks)
    assert all("content" in chunk for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])