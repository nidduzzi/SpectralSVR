import ast
import pathlib
import sqlite3
import sys

session, module_path, *names = sys.argv[1:]
tree = ast.parse(pathlib.Path(module_path).read_text())
definitions = {
    node.name: node
    for node in tree.body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef))
}
kept_lines = [definitions[name] for name in names]
connection = sqlite3.connect(session)
outside = [
    job_id
    for job_id, row in connection.execute(
        "select job_id, start_pos_row from mutation_specs"
    )
    if not any(node.lineno <= row <= node.end_lineno for node in kept_lines)
]
for job_id in outside:
    connection.execute("delete from mutation_specs where job_id = ?", (job_id,))
    connection.execute("delete from work_items where job_id = ?", (job_id,))
connection.commit()
remaining = connection.execute("select count(*) from work_items").fetchone()[0]
print(
    f"{' '.join(names)}: {remaining} mutants kept, {len(outside)} outside them dropped"
)
