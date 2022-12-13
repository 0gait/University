package ast.number;

import ast.ASTNode;

public class ASTUminus extends ASTNumberArithmeticBinaryOperation
{
    public ASTUminus(ASTNode node)
    {
        super(new ASTNum(0), node, NumberArithmeticBinaryOperator.SUB);
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        builder.append("-");
        this.rigth.toString(builder);
        return builder;
    }
}
