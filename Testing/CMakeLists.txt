# download library for testing
SET(TESTING_LIBRARY_URL "http://www.bic.mni.mcgill.ca/~vfonov/temp/beast-testing-1.0.tar.gz")
SET(TESTING_LIBRARY_MD5 "610deb21e80f72a232bbeab1267afe2b")

file(DOWNLOAD "${TESTING_LIBRARY_URL}" "${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0.tar.gz"  EXPECTED_MD5  "${TESTING_LIBRARY_MD5}" SHOW_PROGRESS )

add_custom_target( 
    unpack_beast_testing ALL
    cmake -E tar zxf "${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0.tar.gz" 
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0.tar.gz"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Unpacking beast-testing-1.0.tar.gz" VERBATIM
)

SET(BEAST_TEST_ENVIRONMENT "OMP_NUM_THREADS=1")

MACRO(ADD_BEAST_TEST name cmd)
   ADD_TEST( NAME ${name}
       COMMAND ${cmd} ${ARGV2} ${ARGV3} ${ARGV4} ${ARGV5} ${ARGV6} ${ARGV7} ${ARGV8} ${ARGV9} ${ARGV10} ${ARGV11}
        ${ARGV12} ${ARGV13} ${ARGV14} ${ARGV15} ${ARGV16} ${ARGV17} ${ARGV18} ${ARGV19} ${ARGV20} ${ARGV21} ${ARGV22}
        ${ARGV23} ${ARGV24} ${ARGV25} ${ARGV26}
   )
   set_tests_properties( ${name} PROPERTIES ENVIRONMENT "${BEAST_TEST_ENVIRONMENT}")
   
ENDMACRO(ADD_BEAST_TEST)

ADD_BEAST_TEST(MINCBEAST_V1_SSD_4MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.4mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_4mm.mnc 
  -verbose -clob -median -same_resolution  )

ADD_BEAST_TEST(MINCBEAST_V1_SSD_2MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.2mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_2mm.mnc 
  -verbose -clob -median -same_resolution  )
  
ADD_BEAST_TEST(MINCBEAST_V1_SSD_1MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.1mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_1mm.mnc 
  -verbose -clob -median -same_resolution  )

ADD_BEAST_TEST(MINCBEAST_V1_SPARSE_4MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.4mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_4mm.mnc 
  -verbose -clob -median -same_resolution -sparse  )

ADD_BEAST_TEST(MINCBEAST_V1_SPARSE_2MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.2mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_2mm.mnc 
  -verbose -clob -median -same_resolution -sparse  )
  
ADD_BEAST_TEST(MINCBEAST_V1_SPARSE_1MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.1mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_1mm.mnc 
  -verbose -clob -median -same_resolution -sparse  )

  
ADD_BEAST_TEST(MINCBEAST_V2_SSD_4MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.4mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_4mm.mnc 
  -verbose -clob -median -same_resolution -v2 -selection_num 5  )

ADD_BEAST_TEST(MINCBEAST_V2_SSD_2MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.2mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_2mm.mnc 
  -verbose -clob -median -same_resolution -v2 -selection_num 5)
  
ADD_BEAST_TEST(MINCBEAST_V2_SSD_1MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.1mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_1mm.mnc 
  -verbose -clob -median -same_resolution -v2 -selection_num 5)


ADD_BEAST_TEST(MINCBEAST_V2_SPARSE_4MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.4mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_4mm.mnc 
  -verbose -clob -median -same_resolution -sparse -v2 -selection_num 5)

ADD_BEAST_TEST(MINCBEAST_V2_SPARSE_2MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.2mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_2mm.mnc 
  -verbose -clob -median -same_resolution -sparse -v2 -selection_num 5)
  
ADD_BEAST_TEST(MINCBEAST_V2_SPARSE_1MM ${CMAKE_BINARY_DIR}/mincbeast 
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0
 ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/ICBM_pol/stx/1mm/00100.mnc 
  -conf ${CMAKE_CURRENT_BINARY_DIR}/beast-testing-1.0/default.1mm.conf 
  ${CMAKE_CURRENT_BINARY_DIR}/test_v1_1mm.mnc 
  -verbose -clob -median -same_resolution -sparse -v2 -selection_num 5)

