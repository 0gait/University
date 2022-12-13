package ast.executionFlow.conditionalBranches;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import ast.ASTNodeShortCircuit;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import types.TypeNull;
import types.primitives.TypeBool;
import util.Utils;
import values.IValue;
import values.VNull;
import values.primitive.VBool;

public class ASTIfThenElse extends ASTNodeAbstract
{
    public static final String OPERATOR = "if then else";

    protected ASTNode ifNode, thenNode, elseNode;

    /**
     * @param ifNode
     * @param thenNode
     * @param elseNode
     */
    public ASTIfThenElse(ASTNode ifNode, ASTNode thenNode, ASTNode elseNode) {
        this.ifNode = ifNode;
        this.thenNode = thenNode;
        this.elseNode = elseNode;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e)
    {
        if (this.ifNode instanceof ASTNodeShortCircuit)
            compileWithShortCircuit(c, e);
        else
            compileDefault(c, e);
    }

    public void compileDefault(MainCodeBlock c, Environment<Coordinates> e)
    {
        String l1 = c.getNewLabelId();
        String l2 = c.getNewLabelId();
        
        ifNode.compile(c, e);

        c.emit("ifeq " + l1);
        thenNode.compile(c, e);

        // if we do not have else then the stack height and types must be equal.
        if (this.elseNode == null)
        {
            // remove the value produced by the then execution and push null.
            c.emit("pop");
            c.emitNull();
        }

        c.emit("goto " + l2);
        c.emit(l1 + ": ");

        if (elseNode != null)
            elseNode.compile(c, e);
        else
            c.emitNull();

        c.emit(l2 + ":");
    }

    public void compileWithShortCircuit(MainCodeBlock c, Environment<Coordinates> e) {

        String tl = c.getNewLabelId();
        String fl = c.getNewLabelId();

        String exitLabel = c.getNewLabelId();
        
        ((ASTNodeShortCircuit)ifNode).compile(c, e, tl, fl);

        c.emit(tl + ":");
        thenNode.compile(c, e);

        // if we do not have else then the stack height and types must be equal.
        if (this.elseNode == null)
        {
            // remove the value produced by the then execution and push null.
            c.emit("pop");
            c.emitNull();
        }

        c.emit("goto " + exitLabel );
        c.emit(fl + ":");

        if (this.elseNode != null)
            elseNode.compile(c, e);
        else
            c.emitNull();

        c.emit(exitLabel + ":");
    }

    @Override
    public IValue eval(Environment<IValue> e) {
        boolean value =  Utils.checkValueForOperation(this.ifNode.eval(e), VBool.class, OPERATOR).getValue();

        IValue returnValue = null;
        if (elseNode == null)
            returnValue = VNull.VALUE;

        if(value)
        {
            IValue v = this.thenNode.eval(e);
            return returnValue != null ? returnValue : v;
        }
        else if( elseNode != null)
            return this.elseNode.eval(e);
        
        return returnValue;
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        Utils.checkTypeForOperation(this.ifNode.typecheck(e), TypeBool.class, OPERATOR);

        IType thenType = this.thenNode.typecheck(e);

        if(elseNode != null)
        {
            IType elseType = this.elseNode.typecheck(e);

            if (!thenType.equals(elseType))
                throw new TypeErrorException(String.format("Incompatible types for if statement. The types of" +
                    " the then and else branches must be equal.\nThen branch type: %s\nElse branch type: %s\n",
                    thenType.show(), elseType.show()));

            this.type = thenType;
        }
        else
        {
            /**
             * Because there is no Else branch, the type of this Node is not the one of the Then branch.
             * This makes sure that the result of this construction cannot be used as an expression in other
             * part of the program.
             * This construction can only be used to modify state.
             */
            this.type = TypeNull.TYPE;
        }

        return this.type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder)
    {
        builder.append("if (");
        ((ASTNodeAbstract)(this.ifNode)).toString(builder);
        builder.append(")\nthen\n\t");
        ((ASTNodeAbstract)(this.thenNode)).toString(builder);

        if (this.elseNode != null)
        {
            builder.append("\nelse\n\t");
            ((ASTNodeAbstract)(this.elseNode)).toString(builder);
        }

        builder.append("\nend\n");

        return builder;
    }
}
