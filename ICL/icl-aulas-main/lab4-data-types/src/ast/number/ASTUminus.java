package ast.number;

import ast.ASTNode;

public class ASTUminus extends ASTNumberArithmeticBinaryOperation
{
    public ASTUminus(ASTNode node)
    {
        super(new ASTNum(0), node, NumberArithmeticBinaryOperator.SUB);
    }
}
