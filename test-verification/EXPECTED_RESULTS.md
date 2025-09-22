# Manual Verification Test - Expected Results

This document defines the EXACT expected results for our test verification repository.
Every entity and relationship listed here must be found by our parser.

## Test Repository Structure

```
test-verification/
├── user.py          (48 lines)
├── database.py      (54 lines)
└── EXPECTED_RESULTS.md
```

## Expected Entities (Total: 12)

### File: user.py (7 entities)

1. **CLASS: User** (lines 15-30)
   - Description: "User model with basic validation"
   - Methods: __init__, deactivate

2. **METHOD: User.__init__** (lines 18-23)
   - Description: "Initialize a new user"
   - Parameters: self, user_id, email, name

3. **METHOD: User.deactivate** (lines 25-28)
   - Description: "Deactivate the user account"
   - Parameters: self

4. **FUNCTION: validate_email** (lines 31-34)
   - Description: "Validate email format using regex"
   - Parameters: email
   - Returns: bool

5. **FUNCTION: get_user_by_id** (lines 37-46)
   - Description: "Retrieve user by ID from mock database"
   - Parameters: user_id
   - Returns: Optional[User]

6. **FUNCTION: create_user** (lines 49-59)
   - Description: "Create a new user with validation"
   - Parameters: email, name
   - Returns: User

7. **VARIABLE: mock_users** (lines 39-42)
   - Description: Dictionary with mock user data
   - Type: dict

### File: database.py (5 entities)

1. **CLASS: DatabaseConnection** (lines 13-38)
   - Description: "Simple database connection wrapper"
   - Methods: __init__, connect, execute, close

2. **METHOD: DatabaseConnection.__init__** (lines 16-21)
   - Description: "Initialize database connection"
   - Parameters: self, host, port, database

3. **METHOD: DatabaseConnection.connect** (lines 23-27)
   - Description: "Establish database connection"

4. **METHOD: DatabaseConnection.execute** (lines 29-35)
   - Description: "Execute a database query"
   - Parameters: self, query

5. **METHOD: DatabaseConnection.close** (lines 37-39)
   - Description: "Close the database connection"

6. **FUNCTION: connect_to_database** (lines 42-48)
   - Description: "Create and return a database connection"
   - Parameters: host, port
   - Returns: DatabaseConnection

7. **FUNCTION: execute_query** (lines 51-62)
   - Description: "Execute a query using the provided connection"
   - Parameters: query, connection

## Expected Relationships (Total: 8)

### Import Relationships (1)
1. **database.py IMPORTS user.py**
   - Imports: User, validate_email (line 11)

### Function Call Relationships (5)
1. **create_user CALLS validate_email** (line 52 in user.py)
2. **create_user CALLS User.__init__** (line 58 in user.py)
3. **get_user_by_id CALLS User.__init__** (line 45 in user.py)
4. **connect_to_database CALLS DatabaseConnection.__init__** (line 46 in database.py)
5. **execute_query CALLS validate_email** (line 59 in database.py)

### Class Instantiation Relationships (2)
1. **connect_to_database CREATES DatabaseConnection** (line 46 in database.py)
2. **create_user CREATES User** (line 58 in user.py)

## Validation Criteria

### Entity Extraction Accuracy
- ✅ All 12 entities must be detected
- ✅ Entity types must be correct (CLASS, FUNCTION, METHOD, VARIABLE)
- ✅ Line numbers must be accurate (±1 acceptable for parser differences)
- ✅ Descriptions must be extracted from docstrings
- ✅ Parameters must be identified correctly

### Relationship Extraction Accuracy
- ✅ All 8 relationships must be detected
- ✅ Import relationships must be identified
- ✅ Function calls must be tracked
- ✅ Class instantiations must be detected
- ✅ Cross-file relationships must work

### File Processing
- ✅ Both files must be parsed successfully
- ✅ No parsing errors or exceptions
- ✅ File paths must be relative to repository root

## Test Execution

To verify parser accuracy:

1. **Parse the test repository**:
   ```bash
   ./kgraph parse ./test-verification --reset-repo
   ```

2. **Query for entities**:
   ```bash
   ./kgraph query "User class" --repository-id test-verification
   ./kgraph query "validate_email function" --repository-id test-verification
   ./kgraph query "database connection" --repository-id test-verification
   ```

3. **Verify completeness**:
   - Count total entities: should be exactly 12
   - Count total relationships: should be exactly 8
   - Check each entity type distribution
   - Verify cross-file imports work

## Success Criteria

✅ **100% Entity Detection**: All 12 entities found
✅ **100% Relationship Detection**: All 8 relationships found
✅ **Correct Classification**: Entity types match specification
✅ **Accurate Metadata**: Line numbers, descriptions, parameters correct
✅ **Cross-file Resolution**: Import and call relationships between files work
✅ **Query Functionality**: All entities findable through semantic search

This specification provides a complete, manually verifiable test case for our parser system.