package util;

import java.util.Iterator;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;

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
}
