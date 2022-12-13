package ast.number;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import ast.binaryOperations.ASTArithmeticBinaryOperation;
import ast.binaryOperations.operators.ArithmeticBinaryOperator;

public class ASTUminus extends ASTArithmeticBinaryOperation
{
    public ASTUminus(ASTNode node)
    {
        super(new ASTNum(0), node, ArithmeticBinaryOperator.SUB);
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        builder.append("-");
        ((ASTNodeAbstract)(this.right)).toString(builder);
        return builder;
    }
}
