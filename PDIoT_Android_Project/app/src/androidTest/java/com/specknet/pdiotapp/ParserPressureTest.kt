package com.specknet.pdiotappX1

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*
import java.io.*

@RunWith(AndroidJUnit4::class)
class FileReaderParserIntegrationTest {

    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun testFileReaderParserIntegration() {
        val fileReader = FileReader(appContext)
        val parser = Parser()

        // Test with a valid file
        val validFileContent = fileReader.readFileFromAssets("valid_file.txt")
        val validParseResult = parser.parse(validFileContent)
        assertEquals("Expected result for valid file", validParseResult)

        // Test with a file containing special characters
        val specialCharsContent = fileReader.readFileFromAssets("special_chars_file.txt")
        val specialCharsResult = parser.parse(specialCharsContent)
        assertEquals("Expected result for special characters file", specialCharsResult)

        // Additional tests can be added here for different scenarios
    }
}

class FileReader(val context: Context) {
    fun readFileFromAssets(filename: String): String {
        return context.assets.open(filename).bufferedReader().use { it.readText() }
    }
}

class Parser {
    fun parse(content: String): String {
//        var string: String? = ""
//        val stringBuilder = StringBuilder()
        val file = File(filepath)
        val reader = BufferedReader(FileReader(file))
//        while (true) {
//            try {
//                if (reader.readLine().also { string = it } == null) break
//            } catch (e: IOException) {
//                e.printStackTrace()
//            }
//            stringBuilder.append(string).append("\n")
//        }

        val result: MutableList<List<String>> = ArrayList()
        var line: String?
        try {
            while (reader.readLine().also { line = it } != null) {
                val tokens = line!!.split(",".toRegex()).toTypedArray()
                if (tokens.isNotEmpty()) {
                    result.add(tokens.toList())
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }

        setupTable(result)

        reader.close()
        return parsedContent
    }
}
