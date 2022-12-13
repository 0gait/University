package ast.arithmetic;

import ast.ASTNode;
import compiler.MainCodeBlock;
import util.Environment;
import util.Coordinates;

public class ASTDiv implements ASTNode {

    ASTNode lhs, rhs;

    public int eval(Environment<Integer> e) {
        int v1 = lhs.eval(e);
        int v2 = rhs.eval(e);
        return v1 / v2;
    }

    public ASTDiv(ASTNode l, ASTNode r) {
        lhs = l;
        rhs = r;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        // TODO Auto-generated method stub
        lhs.compile(c, e);
        rhs.compile(c, e);
        c.emit("idiv");
    }
}
