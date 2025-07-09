import ast
import re


def extract_numerator_and_denominators(expr):
    class BinOpVisitor(ast.NodeVisitor):
        def __init__(self):
            self.numerator = None
            self.denominators = []

        def visit_BinOp(self, node):
            if isinstance(node.op, ast.Div) and self.numerator is None:
                self.numerator = ast.unparse(node.left).strip()
                self.process_denominator(node.right)
            else:
                self.generic_visit(node)  # Continue visiting child nodes

        def process_denominator(self, node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                # Split addition into terms
                self.process_denominator(node.left)
                self.process_denominator(node.right)
            else:
                # Extract individual denominator components
                denominators = self.extract_variable_name(ast.unparse(node).strip())
                self.denominators.extend(denominators)

        def extract_variable_name(self, name):
            if '.fillna' in name:
                # Handles expressions like "var.fillna(val)"
                variables = []
                for var in name.split('.fillna'):
                    cleaned_var = var.replace('(', '').replace(')', '').strip()
                    if cleaned_var not in ('', '0'):
                        variables.append(cleaned_var)
                return variables
            try:
                float(name)
                # If conversion to float succeeds, it's a numeric literal; ignore it.
                return []
            except ValueError:
                # Not a number, so it's a symbolic variable name. Extract the base name.
                # e.g., 'app_metric' from 'app_metric', or 'var' from 'var.method'
                base_name = re.sub(r'\..*', '', name)
                return [base_name] if base_name else []

    tree = ast.parse(expr, mode='eval')
    visitor = BinOpVisitor()
    visitor.visit(tree)
    return visitor.numerator, visitor.denominators
