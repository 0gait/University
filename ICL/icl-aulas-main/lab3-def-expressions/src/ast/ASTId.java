package ast;

import compiler.MainCodeBlock;
import util.Coordinates;
import util.Environment;

public class ASTId implements ASTNode{

    protected String id;

    public ASTId(String id){
        this.id = id;
    }
    @Override
    public int eval(Environment<Integer> e) {
        return e.find(id);
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        
        //Frame atual pode nao ser a depth -1. Frame anterior ao atual pode nao ser o atual -1.

        Coordinates coord = e.find(this.id);
        int varFrameId = coord.getLeft();
        String varId = coord.getRight();
        getFieldFromFrame(c, varFrameId, varId);
    }

    /**
     * Emits instructions to get a field from the specified frame ID.
     * 
     * In fact, emits the current frame, then reaches the specified frame
     * from the current one and then emits the instruction to get a field.
     * 
     * @param c - The main code block.
     * @param frameId - The id of the frame.
     * @param fieldId - The id of the field.
     */
    protected void getFieldFromFrame(MainCodeBlock c, int frameId, String fieldId)
    {
        c.emitCurrentFrame();
        c.reachFrameIdFromCurrentFrame(frameId);
        c.emit(String.format("getfield f%d/%s I", frameId, fieldId) );
    }
    
}
