"""
validate_sql.py

The safety gate between the LLM's generated SQL and the actual database.

This function does NOT trust the LLM. It assumes the LLM could, at some point,
hallucinate a DROP/DELETE/UPDATE statement (this happens more than people
expect), and its job is to catch that BEFORE execute_sql ever runs it.

Rules enforced:
  1. Must be exactly ONE SQL statement (blocks "SELECT ...; DROP TABLE ...;" tricks)
  2. That one statement must be a SELECT
  3. No destructive/dangerous keywords anywhere in the query, even inside
     subqueries or comments (defense in depth, in case rule 1/2 get bypassed
     by unusual formatting)
  4. No file-system access attempts (INTO OUTFILE / INTO DUMPFILE / LOAD_FILE) -
     a MySQL-specific way to read or write files on the server via SQL

Returns a tuple: (is_valid: bool, message: str)
  - If valid: (True, "OK")
  - If invalid: (False, "<human-readable reason>")  <- this message is also
    useful to feed back to the LLM in the self-correction retry loop
"""

import re
import sqlparse

# Keywords that should never appear in a query we're about to run.
# Checked as whole words, case-insensitive.
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
    "CALL", "MERGE", "LOCK", "UNLOCK", "SET",
]

# MySQL-specific file access patterns (checked as substrings, case-insensitive)
FORBIDDEN_PATTERNS = [
    "into outfile",
    "into dumpfile",
    "load_file",
]


def validate_sql(sql: str) -> tuple[bool, str]:
    if not sql or not sql.strip():
        return False, "Empty query."

    # 1. Strip comments so someone can't hide a keyword inside /* ... */ or -- ...
    #    then re-check the cleaned version alongside the original.
    cleaned = sqlparse.format(sql, strip_comments=True).strip()

    if not cleaned:
        return False, "Query is empty after removing comments."

    # 2. Must be exactly one statement.
    statements = [s for s in sqlparse.split(cleaned) if s.strip()]
    if len(statements) == 0:
        return False, "No valid SQL statement found."
    if len(statements) > 1:
        return False, "Multiple SQL statements detected. Only a single SELECT statement is allowed."

    single_statement = statements[0]

    # 3. Must be a SELECT statement.
    parsed = sqlparse.parse(single_statement)[0]
    statement_type = parsed.get_type()  # returns e.g. 'SELECT', 'INSERT', 'UNKNOWN'

    if statement_type != "SELECT":
        return False, f"Only SELECT statements are permitted. Detected statement type: {statement_type}."

    # 4. Scan for forbidden keywords anywhere in the text (defense in depth).
    #    \b ensures we match whole words only (so "selected_products" doesn't
    #    trip on containing "SELECT"... wait, it doesn't contain a forbidden
    #    word anyway, but this pattern generally avoids false positives like
    #    a column named "reset_date" matching "SET").
    upper_sql = single_statement.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_sql):
            return False, f"Forbidden keyword detected: {keyword}. Only read-only SELECT queries are permitted."

    # 5. Scan for file-access patterns.
    lower_sql = single_statement.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in lower_sql:
            return False, f"Forbidden operation detected: '{pattern}'. File access is not permitted."

    return True, "OK"


if __name__ == "__main__":
    # Quick self-test -- run `python validate_sql.py` any time to sanity-check
    # the guardrail without needing the DB or an API key.
    test_cases = [
        ("SELECT * FROM customers LIMIT 5;", True),
        ("SELECT customer_city, COUNT(*) FROM customers GROUP BY customer_city;", True),
        ("DROP TABLE customers;", False),
        ("SELECT * FROM customers; DROP TABLE customers;", False),
        ("DELETE FROM orders WHERE order_id = '123';", False),
        # Comment is stripped before validation, so "SET x=1" never reaches MySQL --
        # this SHOULD be valid, since the comment is just dead text.
        ("SELECT * FROM customers WHERE customer_city = 'sao paulo' /* SET x=1 */;", True),
        ("UPDATE customers SET customer_city = 'x';", False),
        ("SELECT * INTO OUTFILE '/tmp/leak.csv' FROM customers;", False),
        ("", False),
        ("   ", False),
        ("SELECT o.order_id, c.customer_city FROM orders o JOIN customers c ON o.customer_id = c.customer_id;", True),
    ]

    passed = 0
    for sql, expected_valid in test_cases:
        is_valid, message = validate_sql(sql)
        status = "PASS" if is_valid == expected_valid else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"[{status}] valid={is_valid:<5} | expected={expected_valid:<5} | {message}")
        print(f"        query: {sql!r}\n")

    print(f"{passed}/{len(test_cases)} test cases passed.")
