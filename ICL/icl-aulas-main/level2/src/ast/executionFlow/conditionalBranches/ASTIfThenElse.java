package ast.executionFlow.conditionalBranches;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.IllegalOperatorException;
import typeError.TypeErrorException;
import types.IType;
import types.primitives.TypeBool;
import values.IValue;
import values.VNull;
import values.primitive.VBool;

public class ASTIfThenElse extends ASTNodeAbstract {
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
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {

        String l1 = c.getNewLabelId();
        String l2 = null;
        
        ifNode.compile(c, e);

        c.emit("ifeq " + l1);
        thenNode.compile(c, e);

        l2 = c.getNewLabelId();
        c.emit("goto " + l2);

        c.emit(l1 + ": ");

        if (elseNode != null)
        {
            elseNode.compile(c, e);
            c.emit(l2 + ":");
        }
        else
        {
            c.emit("sipush 0");
            c.emit(l2 + ":");
        }
    }

    @Override
    public IValue eval(Environment<IValue> e) {
        Boolean value =  checkIfRuntimeType(this.ifNode.eval(e)).getValue();

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
    public IType typecheck(Environment<IType> e) {
        checkTypeIf(this.ifNode.typecheck(e));

        IType thenType = this.thenNode.typecheck(e);

        if(elseNode != null)
        {
            IType elseType = this.elseNode.typecheck(e);

            if (!thenType.equals(elseType))
                throw new TypeErrorException(String.format("Incompatible types for if statement. The types of" +
                    " the then and else branches must be equal.\nThen branch type: %s\nElse branch type: %s\n",
                    thenType.show(), elseType.show()));

            type = thenType;
        }
        else
        {
            /**
             * Because there is no Else branch, the type of this Node is not the one of the Then branch.
             * This makes sure that the result of this construction cannot be used as an expression in other
             * part of the program.
             * This construction can only be used to modify state.
             */

            // TODO: type =
            type = TypeBool.TYPE;
        }

        return thenType;
    }

    protected VBool checkIfRuntimeType(IValue value) {
        boolean checked = value instanceof VBool;

        if (!checked)
            throw new IllegalOperatorException(OPERATOR, TypeBool.TYPE.show(), value.show());

        return (VBool) value;
    }

    protected TypeBool checkTypeIf(IType type) {
        boolean checked = type instanceof TypeBool;

        if (!checked)
            throw new IllegalOperatorException(OPERATOR, TypeBool.TYPE.show(), type.show());

        return (TypeBool) type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        builder.append("if (");
        this.ifNode.toString(builder);
        builder.append(")\nthen\n\t");
        this.thenNode.toString(builder);
        builder.append("\nelse\n\t");
        this.elseNode.toString(builder);
        builder.append("\nend\n");

        return builder;
    }

    

}
