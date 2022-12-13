package ast.names;

import compiler.MainCodeBlock;
import types.IType;
import util.Coordinates;
import util.Environment;
import values.IValue;

public class ASTUp extends ASTId
{

    protected int upDepth;

    /**
     * @param id
     * @param upDepth
     */
    public ASTUp(String id, int upDepth)
    {
        super(id);
        this.upDepth = upDepth;
    }

    @Override
    public IValue eval(Environment<IValue> e) {
        return e.find(id, upDepth);
    }

    @Override
    public IType typecheck(Environment<IType> e) {
        type = e.find(id, upDepth);
        return type;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        //Frame atual pode nao ser a depth -1. Frame anterior ao atual pode nao ser o atual -1.

        Coordinates coord = e.find(this.id, upDepth);
        int varFrameId = coord.getLeft();
        String varId = coord.getRight();
        getFieldFromFrame(c, varFrameId, varId);
    }

}
