

def find_primary_key_relationships(table_names, inner_keys):
    result = []
    # 遍历主键及其对应的表列表
    for primary_key, tables in inner_keys.items():
        related_tables = []
        for table in table_names:
            if table in tables:
                related_tables.append(table)
        if len(related_tables) > 1:
            relationship = "=".join([f"{table}.{primary_key}" for table in related_tables])
            result.append(relationship)
    return result


if __name__ == "__main__":
    table_names = ["table1", "table2", "table3"]
    inner_keys = {"id": ["table1", "table2"], "name": ["table2", "table3"]}
    print(find_primary_key_relationships(table_names, inner_keys))
