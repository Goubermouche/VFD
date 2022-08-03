project "Renderer"
    location "Renderer"
    kind "SharedLib"
    language "C++"

    targetdir "bin/%{cfg.buildcfg}"


    files
    {
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.cpp"
    }



    filter "configurations:Debug"
        defines 
        {
            "DEBUG" 
        }

        links
        {
        }

        symbols "On"
        staticruntime "off"
  
     filter "configurations:Release"
        defines 
        {
            "NDEBUG" 
        }

        links
		{
		
		}
        
        optimize "Full"
        staticruntime "off"