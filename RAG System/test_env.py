from sqlglot import parse_one
from sqlglot.optimizer import optimize
from sqlglot.optimizer.optimize_joins import optimize_joins 
sql = "SELECT * FROM large_table AS lt JOIN small_table AS st ON lt.id = st.id JOIN medium_table AS mt ON mt.id = st.mid"
ast = parse_one(sql)

optimized_ast = optimize(ast, rules=[optimize_joins])
print(optimized_ast.sql())
