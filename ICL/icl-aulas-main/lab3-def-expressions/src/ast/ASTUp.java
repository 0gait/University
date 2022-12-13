package ast;

import compiler.MainCodeBlock;
import util.Coordinates;
import util.Environment;

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
    public int eval(Environment<Integer> e) {
        return e.find(id, upDepth);
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
