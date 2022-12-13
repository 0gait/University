package util;

import java.util.Iterator;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

import typeError.IllegalOperatorException;
import typeError.TypeErrorException;
import typeError.TypeErrorRuntimeException;
import types.IType;
import types.TypeNull;
import values.IValue;
import values.VNull;

public class Utils
{

    public static final String DEFAULT_PRINT_SEPARATOR = ", ";

    public static final String[] DEFAULT_DELIMITERS = {
        "(", ")"
    };

    /**
     * Prints all elements sperated by the specified separator using a string builder.
     * @param <T> - The type of the elements to print.
     * @param elements - The iterable of elements.
     * @param toString - A function to get a String representation of an element.
     *  If null {@link #toString()} is used.
     * @param separator - A separator for the elements or null to use the {@link #DEFAULT_PRINT_SEPARATOR}.
     * @param builder - A String builder to append the result or null.
     * @return A new String builder with the String "(elem0"separator"elem1...)" appended.
     */
    public static <T> StringBuilder toStringList(Iterable<T> elements, Function<T, String> toString,
        String separator, StringBuilder builder)
    {
        return toStringList(elements, toString, separator, DEFAULT_DELIMITERS, builder);
    }

    /**
     * Prints all elements sperated by the specified separator using a string builder.
     * @param <T> - The type of the elements to print.
     * @param elements - The iterable of elements.
     * @param toString - A function to get a String representation of an element.
     *  If null {@link #toString()} is used.
     * @param separator - A separator for the elements or null to use the {@link #DEFAULT_PRINT_SEPARATOR}.
     * @param delimiters - An array with 2 elements to delimit the result or null.
     * @param builder - A String builder to append the result or null.
     * @return A new String builder with the String "(elem0"separator"elem1...)" appended.
     */
    public static <T> StringBuilder toStringList(Iterable<T> elements, Function<T, String> toString,
        String separator, String[] delimiters, StringBuilder builder)
    {
        Objects.requireNonNull(elements);
        
        if (toString == null)
            toString = (elem) -> elem.toString();

        if (separator == null)
            separator = DEFAULT_PRINT_SEPARATOR;

        if (builder == null)
            builder = new StringBuilder();

        if (delimiters != null)
            builder.append(delimiters[0]);

        Iterator<T> iterator = elements.iterator();
        if (iterator.hasNext())
            builder.append(toString.apply(iterator.next()));

        while (iterator.hasNext()) {
            builder.append(separator);
            builder.append(toString.apply(iterator.next()));
        }

        if (delimiters != null)
            builder.append(delimiters[1]);

        return builder;
    }

    /**
     * Prints all elements sperated by the specified separator using a string builder.
     * @param <T> - The type of the elements to print.
     * @param elements - The iterable of elements.
     * @param append - A function that acceps an object and appends a String representation of it to the
     * StringBuilder.
     * @param separator - A separator for the elements or null to use the {@link #DEFAULT_PRINT_SEPARATOR}.
     * @param delimiters - An array with 2 elements to delimit the result or null.
     * @param builder - A String builder to append the result or null.
     * @return A new String builder with the String "(elem0"separator"elem1...)" appended.
     */
    public static <T> StringBuilder toStringList(Iterable<T> elements, Consumer<T> append,
        String separator, String[] delimiters, StringBuilder builder)
    {
        Objects.requireNonNull(elements);
        Objects.requireNonNull(append);

        if (separator == null)
            separator = DEFAULT_PRINT_SEPARATOR;

        if (builder == null)
            builder = new StringBuilder();

        if (delimiters != null)
            builder.append(delimiters[0]);

        Iterator<T> iterator = elements.iterator();
        if (iterator.hasNext())
            append.accept(iterator.next());

        while (iterator.hasNext()) {
            builder.append(separator);
            append.accept(iterator.next());
        }

        if (delimiters != null)
            builder.append(delimiters[1]);

        return builder;
    }

    /**
     * Checks if this value is of type Null.
     * @param value - The value to check.
     * @return - The value if the type is not null.
     */
    public static IValue requireNonNull(IValue value)
    {
        boolean isNull = value instanceof VNull;

        if (isNull)
            throw new TypeErrorRuntimeException("Illegal type.\nType Null cannot be used in any operation.");

        return value;
    }

    /**
     * Checks if this type is Null.
     * @param value - The type to check.
     * @return - The type if it is not null.
     */
    public static IType requireNonNull(IType type) throws TypeErrorException
    {
        boolean isNull = type instanceof TypeNull;

        if (isNull)
            throw new TypeErrorException("Illegal type.\nType Null cannot be used in any operation.");

        return type;
    }

    public static <T extends IValue> T checkValue(IValue value, Class<T> check,
        Supplier<TypeErrorRuntimeException> exception)
    {
        requireNonNull(value);
        boolean checked = check.isInstance(value);

        if (!checked)
            throw exception.get();

        return check.cast(value);
    }


    public static <T extends IType> T checkType(IType type, Class<T> check,
        Supplier<TypeErrorException> exception) throws TypeErrorException
    {
        requireNonNull(type);
        boolean checked = check.isInstance(type);

        if (!checked)
            throw exception.get();

        return check.cast(type);
    }

    public static <T extends IValue> T checkValueForOperation(IValue value, Class<T> check,
        String operator) 
    {
        String expectedType;
        try {
            expectedType = (String)check.getField("TYPE_NAME").get(null);
        } catch (Exception e) {
            throw new Error(e);
        }

        return checkValue(value, check,
            () -> new IllegalOperatorException(operator, expectedType, value.getType().show())
                    .toRuntimeException());
    }

    public static <T extends IType> T checkTypeForOperation(IType type, Class<T> check,
        String operator) throws TypeErrorException
    {
        String expectedType;
        try {
            expectedType = (String)check.getField("TYPE_NAME").get(null);
        } catch (Exception e) {
            throw new Error(e);
        }

        return checkType(type, check,
            () -> new IllegalOperatorException(operator, expectedType, type.show()));
    }

    public static <T extends IValue> T checkValueForOperation(IValue value, Class<T> check,
        String operator, String message)
    {
        String expectedType;
        try {
            expectedType = (String)check.getField("TYPE_NAME").get(null);
        } catch (Exception e) {
            throw new Error(e);
        }

        return checkValue(value, check,
            () -> new IllegalOperatorException(message, operator, expectedType, value.getType().show())
                    .toRuntimeException());
    }

    public static <T extends IType> T checkTypeForOperation(IType type, Class<T> check,
            String operator, String message) throws TypeErrorException
    {
        String expectedType;
        try {
            expectedType = (String) check.getField("TYPE_NAME").get(null);
        } catch (Exception e) {
            throw new Error(e);
        }

        return checkType(type, check,
                () -> new IllegalOperatorException(message, operator, expectedType, type.show()));
    }
}
